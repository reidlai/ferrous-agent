use std::sync::Arc;

use rig_core::tool::{args_schema, Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::client::FerrousApiClient;
use crate::models::OrderDetails;
use crate::tools::ToolError;

/// Arguments for [`ApiGetOrderTool`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetOrderArgs {
    /// Unique order identifier (e.g., `ORD-1001`).
    pub order_id: String,
}

/// `rig_core::tool::Tool` calling `GET /api/v1/orders/{order_id}` on `ferrous-api`.
#[derive(Clone, Debug)]
pub struct ApiGetOrderTool {
    client: Arc<FerrousApiClient>,
}

impl ApiGetOrderTool {
    pub fn new(client: Arc<FerrousApiClient>) -> Self {
        Self { client }
    }
}

impl Tool for ApiGetOrderTool {
    const NAME: &'static str = "api_get_order_details";
    type Args = GetOrderArgs;
    type Output = OrderDetails;
    type Error = ToolError;

    fn description(&self) -> String {
        "Fetch order status, line items, freight cost, and delay metadata from ferrous-api (GET /api/v1/orders/{order_id}).".to_string()
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
            .get_order(&args.order_id)
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
    async fn api_get_order_tool_calls_endpoint_with_bearer_auth() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = r#"{"order_id":"ORD-77","customer_id":"C-9","status":"IN_TRANSIT","total_amount_cents":5000,"currency":"BRL"}"#;

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

        let client = Arc::new(FerrousApiClient::new(
            format!("http://{addr}"),
            "kanidm-token-77",
        ));
        let tool = ApiGetOrderTool::new(client);
        let mut ctx = ToolContext::new();
        let out = tool
            .call(
                &mut ctx,
                GetOrderArgs {
                    order_id: "ORD-77".to_string(),
                },
            )
            .await
            .expect("tool call should succeed");

        assert_eq!(out.order_id, "ORD-77");
        assert_eq!(out.status, "IN_TRANSIT");
        let req = server.await.unwrap();
        assert!(req.starts_with("GET /api/v1/orders/ORD-77 HTTP/1.1"));
        assert!(req
            .to_ascii_lowercase()
            .contains("authorization: bearer kanidm-token-77"));
    }
}
