use std::sync::Arc;

use rig_core::tool::{args_schema, Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::client::FerrousApiClient;
use crate::models::RagHit;
use crate::tools::ToolError;

/// Shared Qdrant RAG query arguments for [`PolicyRagTool`] and [`TranscriptsRagTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RagQueryArgs {
    /// Natural-language search query.
    pub query: String,
    /// Optional precomputed embedding vector.
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    /// Maximum number of hits to return (1..=50, default 5).
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Shared Qdrant RAG query result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RagQueryOutput {
    pub collection: String,
    pub hits: Vec<RagHit>,
}

/// Shared helper executing a Qdrant collection search for RAG tools.
pub(crate) async fn execute_rag_search(
    client: &FerrousApiClient,
    collection: &str,
    args: RagQueryArgs,
) -> Result<RagQueryOutput, ToolError> {
    let limit = args.limit.unwrap_or(5);
    let hits = client
        .search_qdrant(collection, &args.query, args.vector, limit)
        .await
        .map_err(ToolError::from)?;
    Ok(RagQueryOutput {
        collection: collection.to_string(),
        hits,
    })
}

/// `rig_core::tool::Tool` querying Qdrant `/collections/cx_policy_knowledge/points/search`.
#[derive(Clone, Debug)]
pub struct PolicyRagTool {
    client: Arc<FerrousApiClient>,
}

impl PolicyRagTool {
    pub const COLLECTION: &'static str = "cx_policy_knowledge";

    pub fn new(client: Arc<FerrousApiClient>) -> Self {
        Self { client }
    }
}

impl Tool for PolicyRagTool {
    const NAME: &'static str = "policy_rag";
    type Args = RagQueryArgs;
    type Output = RagQueryOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Search CX policy SOPs and courtesy credit eligibility rules in Qdrant collection `cx_policy_knowledge`.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        execute_rag_search(&self.client, Self::COLLECTION, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn policy_rag_tool_queries_cx_policy_knowledge_and_rejects_empty_query() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = r#"{"result":[{"id":1,"score":0.92,"payload":{"sop":"SOP-CC-01","rule":"Up to $20.00 credit for >=3 days delay"}}]}"#;

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
            "qdrant-tok",
        ));
        let tool = PolicyRagTool::new(client);
        let mut ctx = ToolContext::new();

        let err = tool
            .call(
                &mut ctx,
                RagQueryArgs {
                    query: "   ".to_string(),
                    vector: None,
                    limit: None,
                },
            )
            .await
            .expect_err("empty query must fail");
        assert!(matches!(err, ToolError::InvalidInput(_)));

        let out = tool
            .call(
                &mut ctx,
                RagQueryArgs {
                    query: "courtesy credit delay threshold".to_string(),
                    vector: Some(vec![0.1, 0.2, 0.3]),
                    limit: Some(3),
                },
            )
            .await
            .expect("policy_rag should succeed");
        assert_eq!(out.collection, "cx_policy_knowledge");
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].payload["sop"], "SOP-CC-01");

        let req = server.await.unwrap();
        assert!(req.starts_with("POST /collections/cx_policy_knowledge/points/search HTTP/1.1"));
        assert!(req
            .to_ascii_lowercase()
            .contains("authorization: bearer qdrant-tok"));
    }
}
