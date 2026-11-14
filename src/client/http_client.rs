//! Authenticated `reqwest` HTTP client for `ferrous-api` and Qdrant REST endpoints.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use crate::config::SecretString;
use crate::models::{OrderDetails, RagHit, RagSearchResponse, TrackingDetails};
use crate::telemetry::inject_w3c_trace_context;

/// Default Qdrant embedding vector dimension when a text-only query is issued without a caller vector.
const DEFAULT_FALLBACK_VECTOR_DIM: usize = 8;

/// Errors returned by [`FerrousApiClient`].
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("missing required configuration: {0}")]
    MissingConfig(&'static str),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API request to {endpoint} failed with HTTP {status}: {body}")]
    ApiStatus {
        status: u16,
        endpoint: String,
        body: String,
    },

    #[error("failed to decode response from {endpoint}: {message}")]
    Decode { endpoint: String, message: String },
}

#[derive(Serialize)]
struct QdrantSearchRequest<'a> {
    vector: Vec<f32>,
    limit: usize,
    with_payload: bool,
    filter: QdrantTextFilter<'a>,
}

#[derive(Serialize)]
struct QdrantTextFilter<'a> {
    #[serde(rename = "query_text")]
    query_text: &'a str,
}

/// Authenticated REST client for `ferrous-api` and Qdrant, attaching Kanidm M2M `Bearer` tokens.
#[derive(Clone)]
pub struct FerrousApiClient {
    http: reqwest::Client,
    api_base_url: String,
    qdrant_base_url: String,
    auth_token: SecretString,
}

impl fmt::Debug for FerrousApiClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FerrousApiClient")
            .field("api_base_url", &self.api_base_url)
            .field("qdrant_base_url", &self.qdrant_base_url)
            .field("auth_token", &self.auth_token)
            .finish()
    }
}

impl FerrousApiClient {
    /// Construct a [`FerrousApiClient`] from explicit base URL and Kanidm M2M token.
    pub fn new(api_base_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        let api_base_url = normalize_base_url(&api_base_url.into());
        let qdrant_base_url = api_base_url.clone();
        let token = strip_bearer_prefix(&auth_token.into());

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            http,
            api_base_url,
            qdrant_base_url,
            auth_token: SecretString::new(token),
        }
    }

    /// Override the Qdrant base URL (defaults to `api_base_url` if unset).
    pub fn with_qdrant_url(mut self, qdrant_base_url: impl Into<String>) -> Self {
        self.qdrant_base_url = normalize_base_url(&qdrant_base_url.into());
        self
    }

    /// Construct from process environment (`FERROUS_API_URL`, `FERROUS_AUTH_TOKEN`, optional `QDRANT_URL`).
    pub fn from_env() -> Result<Self, ClientError> {
        let env: HashMap<String, String> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Self::from_map(&env)
    }

    /// Construct from an explicit environment map (for hermetic testing).
    pub fn from_map(env: &HashMap<String, String>) -> Result<Self, ClientError> {
        let api_url = env
            .get("FERROUS_API_URL")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or(ClientError::MissingConfig("FERROUS_API_URL"))?;
        let token = env
            .get("FERROUS_AUTH_TOKEN")
            .map(|s| s.trim())
            .map(strip_bearer_prefix)
            .filter(|s| !s.is_empty())
            .ok_or(ClientError::MissingConfig("FERROUS_AUTH_TOKEN"))?;

        let mut client = Self::new(api_url, token);
        if let Some(qdrant_url) = env
            .get("QDRANT_URL")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            client = client.with_qdrant_url(qdrant_url);
        }
        Ok(client)
    }

    /// Configured `ferrous-api` base URL.
    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    /// Configured Qdrant base URL.
    pub fn qdrant_base_url(&self) -> &str {
        &self.qdrant_base_url
    }

    /// Call `GET /api/v1/orders/{order_id}` with `Authorization: Bearer <token>`.
    pub async fn get_order(&self, order_id: &str) -> Result<OrderDetails, ClientError> {
        let id = validate_order_id(order_id)?;
        let endpoint = format!("/api/v1/orders/{id}");
        let url = format!("{}{}", self.api_base_url, endpoint);
        self.get_json(&url, &endpoint).await
    }

    /// Call `GET /api/v1/orders/{order_id}/tracking` with `Authorization: Bearer <token>`.
    pub async fn get_tracking(&self, order_id: &str) -> Result<TrackingDetails, ClientError> {
        let id = validate_order_id(order_id)?;
        let endpoint = format!("/api/v1/orders/{id}/tracking");
        let url = format!("{}{}", self.api_base_url, endpoint);
        self.get_json(&url, &endpoint).await
    }

    /// Call `POST /collections/{collection}/points/search` against Qdrant.
    pub async fn search_qdrant(
        &self,
        collection: &str,
        query_text: &str,
        vector: Option<Vec<f32>>,
        limit: usize,
    ) -> Result<Vec<RagHit>, ClientError> {
        let collection = validate_path_segment(collection, "collection")?;

        let query_text = query_text.trim();
        if query_text.is_empty() && vector.as_ref().is_none_or(Vec::is_empty) {
            return Err(ClientError::InvalidInput(
                "query_text or non-empty vector is required".to_string(),
            ));
        }

        let limit = limit.clamp(1, 50);
        let vec_payload = vector
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![0.0_f32; DEFAULT_FALLBACK_VECTOR_DIM]);

        let endpoint = format!("/collections/{collection}/points/search");
        let url = format!("{}{}", self.qdrant_base_url, endpoint);
        let body = QdrantSearchRequest {
            vector: vec_payload,
            limit,
            with_payload: true,
            filter: QdrantTextFilter { query_text },
        };

        let mut trace_headers = reqwest::header::HeaderMap::new();
        inject_w3c_trace_context(&mut trace_headers);
        let mut req = self.http.post(&url).headers(trace_headers).json(&body);
        if !self.auth_token.expose().is_empty() {
            req = req.bearer_auth(self.auth_token.expose());
        }

        let resp = req.send().await?;
        let envelope: RagSearchResponse = decode_response(resp, &endpoint).await?;
        Ok(envelope.result)
    }

    async fn get_json<T: DeserializeOwned>(
        &self,
        url: &str,
        endpoint: &str,
    ) -> Result<T, ClientError> {
        let mut trace_headers = reqwest::header::HeaderMap::new();
        inject_w3c_trace_context(&mut trace_headers);
        let mut req = self.http.get(url).headers(trace_headers);
        if !self.auth_token.expose().is_empty() {
            req = req.bearer_auth(self.auth_token.expose());
        }
        let resp = req.send().await?;
        decode_response(resp, endpoint).await
    }
}

async fn decode_response<T: DeserializeOwned>(
    resp: reqwest::Response,
    endpoint: &str,
) -> Result<T, ClientError> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(ClientError::ApiStatus {
            status: status.as_u16(),
            endpoint: endpoint.to_string(),
            body,
        });
    }
    let text = resp.text().await?;
    serde_json::from_str::<T>(&text).map_err(|e| ClientError::Decode {
        endpoint: endpoint.to_string(),
        message: e.to_string(),
    })
}

fn validate_order_id(order_id: &str) -> Result<&str, ClientError> {
    validate_path_segment(order_id, "order_id")
}

fn validate_path_segment<'a>(value: &'a str, field: &str) -> Result<&'a str, ClientError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ClientError::InvalidInput(format!(
            "{field} must not be empty"
        )));
    }
    if trimmed == "."
        || trimmed == ".."
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains('?')
        || trimmed.contains('#')
        || trimmed.chars().any(char::is_whitespace)
    {
        return Err(ClientError::InvalidInput(format!(
            "{field} '{trimmed}' contains invalid path characters"
        )));
    }
    Ok(trimmed)
}

fn normalize_base_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn strip_bearer_prefix(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("bearer") {
        String::new()
    } else if trimmed
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("bearer "))
    {
        trimmed[7..].trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_mock_http(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let req_text = String::from_utf8_lossy(&buf[..n]).to_string();

            let response = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            req_text
        });
        (format!("http://{addr}"), handle)
    }

    #[test]
    fn debug_redacts_auth_token_and_strips_bearer_prefix() {
        let secret = "kanidm-m2m-secret-999";
        let client = FerrousApiClient::new("http://localhost:8080/", format!("Bearer {secret}"));
        assert_eq!(client.api_base_url(), "http://localhost:8080");
        assert_eq!(client.auth_token.expose(), secret);

        let debug = format!("{client:?}");
        assert!(!debug.contains(secret));
        assert!(debug.contains("[REDACTED]"));
    }

    #[tokio::test]
    async fn get_order_sends_bearer_token_and_decodes_response() {
        let body = r#"{"order_id":"ORD-42","customer_id":"CUST-1","status":"DELAYED","total_amount_cents":12500,"currency":"BRL","days_delayed":5,"freight_cents":1900}"#;
        let (base_url, handle) = spawn_mock_http("200 OK", body).await;
        let client = FerrousApiClient::new(base_url, "m2m-token-abc");

        let order = client
            .get_order("ORD-42")
            .await
            .expect("get_order should succeed");
        assert_eq!(order.order_id, "ORD-42");
        assert_eq!(order.status, "DELAYED");
        assert_eq!(order.days_delayed, Some(5));

        let raw_req = handle.await.unwrap();
        assert!(raw_req.starts_with("GET /api/v1/orders/ORD-42 HTTP/1.1"));
        assert!(raw_req
            .to_ascii_lowercase()
            .contains("authorization: bearer m2m-token-abc"));
    }

    #[tokio::test]
    async fn api_error_status_does_not_leak_auth_token() {
        let secret = "super-secret-kanidm-token";
        let (base_url, _handle) =
            spawn_mock_http("404 Not Found", r#"{"error":"not found"}"#).await;
        let client = FerrousApiClient::new(base_url, secret);

        let err = client
            .get_order("ORD-MISSING")
            .await
            .expect_err("404 must return ApiStatus error");
        assert!(matches!(err, ClientError::ApiStatus { status: 404, .. }));
        assert!(!err.to_string().contains(secret));
        assert!(!format!("{err:?}").contains(secret));
    }

    #[tokio::test]
    async fn invalid_order_id_and_collection_rejected_before_http_call() {
        let client = FerrousApiClient::new("http://127.0.0.1:9", "tok");
        for bad in ["", "   ", ".", "..", "ORD/1", "ORD\\1", "ORD?x=1", "ORD 1"] {
            let err = client
                .get_order(bad)
                .await
                .expect_err("invalid order_id should fail");
            assert!(matches!(err, ClientError::InvalidInput(_)));

            let q_err = client
                .search_qdrant(bad, "delay policy", None, 5)
                .await
                .expect_err("invalid collection should fail");
            assert!(matches!(q_err, ClientError::InvalidInput(_)));
        }
    }

    #[test]
    fn strip_bearer_prefix_handles_multibyte_utf8_without_panicking() {
        assert_eq!(strip_bearer_prefix("ééééééé"), "ééééééé");
        assert_eq!(strip_bearer_prefix("Bearer 🔑tok"), "🔑tok");
        assert_eq!(strip_bearer_prefix("bearer   "), "");
    }

    #[test]
    fn from_map_validates_required_fields_and_rejects_empty_bearer_token() {
        let mut env = HashMap::new();
        assert!(matches!(
            FerrousApiClient::from_map(&env),
            Err(ClientError::MissingConfig("FERROUS_API_URL"))
        ));

        env.insert(
            "FERROUS_API_URL".to_string(),
            "http://localhost:8080/".to_string(),
        );
        env.insert("FERROUS_AUTH_TOKEN".to_string(), "Bearer   ".to_string());
        assert!(matches!(
            FerrousApiClient::from_map(&env),
            Err(ClientError::MissingConfig("FERROUS_AUTH_TOKEN"))
        ));

        env.insert(
            "FERROUS_AUTH_TOKEN".to_string(),
            "Bearer valid-tok".to_string(),
        );
        env.insert("QDRANT_URL".to_string(), "http://qdrant:6333/".to_string());
        let client = FerrousApiClient::from_map(&env).expect("valid map should succeed");
        assert_eq!(client.api_base_url(), "http://localhost:8080");
        assert_eq!(client.qdrant_base_url(), "http://qdrant:6333");
        assert_eq!(client.auth_token.expose(), "valid-tok");
    }

    #[tokio::test]
    async fn with_qdrant_url_routes_search_qdrant_and_sends_expected_json_body() {
        let body = r#"{"result":[]}"#;
        let (qdrant_url, handle) = spawn_mock_http("200 OK", body).await;
        let client =
            FerrousApiClient::new("http://127.0.0.1:9", "qdrant-m2m").with_qdrant_url(&qdrant_url);

        let hits = client
            .search_qdrant(
                "cx_policy_knowledge",
                "delay > 4 days",
                Some(vec![0.25, 0.75]),
                3,
            )
            .await
            .expect("search_qdrant should route to qdrant_url");
        assert!(hits.is_empty());

        let raw_req = handle.await.unwrap();
        assert!(raw_req.starts_with("POST /collections/cx_policy_knowledge/points/search HTTP/1.1"));
        let (_, json_str) = raw_req.split_once("\r\n\r\n").expect("HTTP body");
        let sent: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON body");
        assert_eq!(sent["vector"], serde_json::json!([0.25, 0.75]));
        assert_eq!(sent["limit"], 3);
        assert_eq!(sent["with_payload"], true);
        assert_eq!(sent["filter"]["query_text"], "delay > 4 days");
    }
}
