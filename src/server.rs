//! HTTP service router, `/health` readiness endpoint, and `--healthcheck` probe client.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::benchmark::{benchmark_sse_router, BenchmarkSseHub};

/// Default timeout for outbound `/health` and Qdrant `/readyz` probes.
pub const DEFAULT_HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Shared readiness state for `GET /health`.
#[derive(Clone, Debug)]
pub struct HealthState {
    ready: Arc<AtomicBool>,
    qdrant_url: Option<String>,
    http: reqwest::Client,
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new(true, None)
    }
}

impl HealthState {
    /// Construct a [`HealthState`] with explicit agent readiness and optional Qdrant base URL.
    pub fn new(ready: bool, qdrant_url: Option<String>) -> Self {
        let qdrant_url = qdrant_url
            .map(|u| u.trim().trim_end_matches('/').to_string())
            .filter(|u| !u.is_empty());

        let http = reqwest::Client::builder()
            .timeout(DEFAULT_HEALTH_PROBE_TIMEOUT)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            ready: Arc::new(AtomicBool::new(ready)),
            qdrant_url,
            http,
        }
    }

    /// Construct a [`HealthState`] reading `QDRANT_URL` from the process environment.
    pub fn from_env() -> Self {
        let qdrant_url = std::env::var("QDRANT_URL").ok();
        Self::new(true, qdrant_url)
    }

    /// Return whether the agent itself is marked ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Update the agent readiness flag (for example, to `false` during graceful shutdown).
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    /// Return the normalized `QDRANT_URL`, if configured.
    pub fn qdrant_url(&self) -> Option<&str> {
        self.qdrant_url.as_deref()
    }
}

/// JSON payload returned by `GET /health`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub agent_ready: bool,
    pub qdrant: String,
}

/// Build the production Axum [`Router`] exposing `GET /health` and benchmark SSE streaming.
pub fn service_router(health_state: HealthState, sse_hub: BenchmarkSseHub) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .with_state(health_state)
        .merge(benchmark_sse_router(sse_hub))
}

async fn health_handler(State(state): State<HealthState>) -> impl IntoResponse {
    if !state.is_ready() {
        let qdrant_status = if state.qdrant_url.is_some() {
            "skipped"
        } else {
            "unconfigured"
        };
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "unavailable".to_string(),
                agent_ready: false,
                qdrant: qdrant_status.to_string(),
            }),
        );
    }

    let Some(qdrant_url) = state.qdrant_url.as_deref() else {
        return (
            StatusCode::OK,
            Json(HealthResponse {
                status: "ok".to_string(),
                agent_ready: true,
                qdrant: "unconfigured".to_string(),
            }),
        );
    };

    let readyz_url = format!("{qdrant_url}/readyz");
    match state.http.get(&readyz_url).send().await {
        Ok(resp) if resp.status().is_success() => (
            StatusCode::OK,
            Json(HealthResponse {
                status: "ok".to_string(),
                agent_ready: true,
                qdrant: "ok".to_string(),
            }),
        ),
        Ok(resp) => {
            tracing::warn!(
                url = %readyz_url,
                status = %resp.status(),
                "Qdrant readiness probe returned non-success status"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(HealthResponse {
                    status: "unavailable".to_string(),
                    agent_ready: true,
                    qdrant: "unavailable".to_string(),
                }),
            )
        }
        Err(err) => {
            tracing::warn!(
                url = %readyz_url,
                error = %err,
                "Qdrant readiness probe failed"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(HealthResponse {
                    status: "unavailable".to_string(),
                    agent_ready: true,
                    qdrant: "unavailable".to_string(),
                }),
            )
        }
    }
}

/// Probe a healthcheck URL (used by `ferrous-agent --healthcheck`).
///
/// Returns `Ok(())` when the endpoint responds with a 2xx HTTP status code,
/// or an error description otherwise.
pub async fn probe_health(url: &str) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|err| format!("failed to build healthcheck HTTP client: {err}"))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("healthcheck request to {url} failed: {err}"))?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!(
            "healthcheck endpoint {url} returned HTTP {}",
            resp.status()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    async fn spawn_test_server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral test port");
        let addr = listener.local_addr().expect("local addr");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn health_returns_200_when_ready_and_qdrant_unconfigured() {
        let app = service_router(HealthState::new(true, None), BenchmarkSseHub::new());
        let (base_url, handle) = spawn_test_server(app).await;

        let resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let body: HealthResponse = resp.json().await.expect("parse HealthResponse");
        assert_eq!(
            body,
            HealthResponse {
                status: "ok".to_string(),
                agent_ready: true,
                qdrant: "unconfigured".to_string(),
            }
        );

        handle.abort();
    }

    #[tokio::test]
    async fn health_returns_503_when_agent_not_ready() {
        let app = service_router(HealthState::new(false, None), BenchmarkSseHub::new());
        let (base_url, handle) = spawn_test_server(app).await;

        let resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(resp.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

        let body: HealthResponse = resp.json().await.expect("parse HealthResponse");
        assert!(!body.agent_ready);
        assert_eq!(body.status, "unavailable");

        handle.abort();
    }

    #[tokio::test]
    async fn health_returns_200_when_mock_qdrant_readyz_succeeds() {
        let mock_qdrant = Router::new().route("/readyz", get(|| async { StatusCode::OK }));
        let (qdrant_url, qdrant_handle) = spawn_test_server(mock_qdrant).await;

        let app = service_router(
            HealthState::new(true, Some(qdrant_url)),
            BenchmarkSseHub::new(),
        );
        let (base_url, handle) = spawn_test_server(app).await;

        let resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let body: HealthResponse = resp.json().await.expect("parse HealthResponse");
        assert_eq!(body.qdrant, "ok");
        assert_eq!(body.status, "ok");

        handle.abort();
        qdrant_handle.abort();
    }

    #[tokio::test]
    async fn health_returns_503_when_qdrant_is_unreachable() {
        // Bind and immediately drop a listener to guarantee an unreachable port.
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let closed_addr = listener.local_addr().expect("addr");
        drop(listener);

        let app = service_router(
            HealthState::new(true, Some(format!("http://{closed_addr}"))),
            BenchmarkSseHub::new(),
        );
        let (base_url, handle) = spawn_test_server(app).await;

        let resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(resp.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

        let body: HealthResponse = resp.json().await.expect("parse HealthResponse");
        assert_eq!(body.qdrant, "unavailable");
        assert_eq!(body.status, "unavailable");

        handle.abort();
    }

    #[tokio::test]
    async fn probe_health_succeeds_on_2xx_and_fails_on_503_or_unreachable() {
        let app = service_router(HealthState::new(true, None), BenchmarkSseHub::new());
        let (base_url, handle) = spawn_test_server(app).await;
        probe_health(&format!("{base_url}/health"))
            .await
            .expect("probe_health should succeed on 200 OK");
        handle.abort();

        let unhealthy_app = service_router(HealthState::new(false, None), BenchmarkSseHub::new());
        let (unhealthy_url, unhealthy_handle) = spawn_test_server(unhealthy_app).await;
        let err = probe_health(&format!("{unhealthy_url}/health"))
            .await
            .expect_err("probe_health should fail on 503");
        assert!(err.contains("503"));
        unhealthy_handle.abort();

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let closed_addr = listener.local_addr().expect("addr");
        drop(listener);
        probe_health(&format!("http://{closed_addr}/health"))
            .await
            .expect_err("probe_health should fail on unreachable port");
    }

    #[tokio::test]
    async fn service_router_merges_benchmark_sse_route() {
        let hub = BenchmarkSseHub::new();
        let app = service_router(HealthState::new(true, None), hub);
        let (base_url, handle) = spawn_test_server(app).await;

        // Unknown run_id returns 404 from benchmark_sse_router (proving the route is mounted).
        let resp = reqwest::get(format!("{base_url}/api/v1/benchmarks/unknown-run/stream"))
            .await
            .expect("GET SSE route");
        assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

        handle.abort();
    }

    #[tokio::test]
    async fn health_returns_503_when_mock_qdrant_readyz_returns_503() {
        let mock_qdrant =
            Router::new().route("/readyz", get(|| async { StatusCode::SERVICE_UNAVAILABLE }));
        let (qdrant_url, qdrant_handle) = spawn_test_server(mock_qdrant).await;

        let app = service_router(
            HealthState::new(true, Some(qdrant_url)),
            BenchmarkSseHub::new(),
        );
        let (base_url, handle) = spawn_test_server(app).await;

        let resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(resp.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

        let body: HealthResponse = resp.json().await.expect("parse HealthResponse");
        assert_eq!(
            body,
            HealthResponse {
                status: "unavailable".to_string(),
                agent_ready: true,
                qdrant: "unavailable".to_string(),
            }
        );

        handle.abort();
        qdrant_handle.abort();
    }

    #[test]
    fn health_state_normalizes_qdrant_url() {
        assert_eq!(
            HealthState::new(true, Some(String::new())).qdrant_url(),
            None
        );
        assert_eq!(
            HealthState::new(true, Some("   ".to_string())).qdrant_url(),
            None
        );
        assert_eq!(
            HealthState::new(true, Some("http://127.0.0.1:6333/".to_string())).qdrant_url(),
            Some("http://127.0.0.1:6333")
        );
    }

    #[tokio::test]
    async fn health_state_set_ready_flips_endpoint_to_503() {
        let state = HealthState::new(true, None);
        let app = service_router(state.clone(), BenchmarkSseHub::new());
        let (base_url, handle) = spawn_test_server(app).await;

        let ok_resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health");
        assert_eq!(ok_resp.status(), reqwest::StatusCode::OK);

        state.set_ready(false);
        assert!(!state.is_ready());

        let unavail_resp = reqwest::get(format!("{base_url}/health"))
            .await
            .expect("GET /health after set_ready(false)");
        assert_eq!(
            unavail_resp.status(),
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );

        handle.abort();
    }
}
