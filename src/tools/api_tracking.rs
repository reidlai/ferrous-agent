use std::sync::Arc;

use rig_core::tool::{args_schema, Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::client::FerrousApiClient;
use crate::models::TrackingDetails;
use crate::tools::ToolError;

/// Arguments for [`ApiGetTrackingTool`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetTrackingArgs {
    /// Unique order identifier (e.g., `ORD-1001`).
    pub order_id: String,
}

/// `rig_core::tool::Tool` calling `GET /api/v1/orders/{order_id}/tracking` on `ferrous-api`.
#[derive(Clone, Debug)]
pub struct ApiGetTrackingTool {
    client: Arc<FerrousApiClient>,
}

impl ApiGetTrackingTool {
    pub fn new(client: Arc<FerrousApiClient>) -> Self {
        Self { client }
    }
}

impl Tool for ApiGetTrackingTool {
    const NAME: &'static str = "api_get_tracking";
    type Args = GetTrackingArgs;
    type Output = TrackingDetails;
    type Error = ToolError;

    fn description(&self) -> String {
        "Fetch shipment tracking milestones, carrier SLA status, and delay days from ferrous-api (GET /api/v1/orders/{order_id}/tracking).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.client
            .get_tracking(&args.order_id)
            .await
            .map_err(ToolError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn api_get_tracking_tool_calls_tracking_endpoint_and_rejects_empty_id() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = r#"{"order_id":"ORD-88","carrier":"Correios SEDEX","current_status":"DELAYED","days_delayed":5,"sla_breached":true,"milestones":[{"timestamp":"2026-11-10T10:00:00Z","status":"IN_TRANSIT","location":"Sao Paulo"}]}"#;

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = socket.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            req
        });

        let client = Arc::new(FerrousApiClient::new(format!("http://{addr}"), "tok-88"));
        let tool = ApiGetTrackingTool::new(client);
        let mut ctx = ToolContext::new();

        let err = tool
            .call(
                &mut ctx,
                GetTrackingArgs {
                    order_id: "   ".to_string(),
                },
            )
            .await
            .expect_err("empty order_id must fail before HTTP");
        assert!(matches!(err, ToolError::InvalidInput(_)));

        let out = tool
            .call(
                &mut ctx,
                GetTrackingArgs {
                    order_id: "ORD-88".to_string(),
                },
            )
            .await
            .expect("tracking tool should succeed");
        assert_eq!(out.carrier, "Correios SEDEX");
        assert_eq!(out.days_delayed, 5);
        assert!(out.sla_breached);

        let req = server.await.unwrap();
        assert!(req.starts_with("GET /api/v1/orders/ORD-88/tracking HTTP/1.1"));
        assert!(req
            .to_ascii_lowercase()
            .contains("authorization: bearer tok-88"));
    }
}
