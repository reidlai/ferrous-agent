use std::sync::Arc;

use rig_core::tool::{args_schema, Tool, ToolContext};

use crate::client::FerrousApiClient;
use crate::tools::policy_rag::{execute_rag_search, RagQueryArgs, RagQueryOutput};
use crate::tools::ToolError;

/// `rig_core::tool::Tool` querying Qdrant `/collections/support_transcripts/points/search`.
#[derive(Clone, Debug)]
pub struct TranscriptsRagTool {
    client: Arc<FerrousApiClient>,
}

impl TranscriptsRagTool {
    pub const COLLECTION: &'static str = "support_transcripts";

    pub fn new(client: Arc<FerrousApiClient>) -> Self {
        Self { client }
    }
}

impl Tool for TranscriptsRagTool {
    const NAME: &'static str = "transcripts_rag";
    type Args = RagQueryArgs;
    type Output = RagQueryOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Search prior support ticket transcripts and dispute precedents in Qdrant collection `support_transcripts`.".to_string()
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
    async fn transcripts_rag_tool_queries_support_transcripts() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = r#"{"result":[{"id":"tx-10","score":0.87,"payload":{"resolution":"Issued $10 courtesy credit"}}]}"#;

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
        let tool = TranscriptsRagTool::new(client);
        let mut ctx = ToolContext::new();

        let out = tool
            .call(
                &mut ctx,
                RagQueryArgs {
                    query: "Correios SEDEX delayed 5 days".to_string(),
                    vector: None,
                    limit: Some(2),
                },
            )
            .await
            .expect("transcripts_rag should succeed");
        assert_eq!(out.collection, "support_transcripts");
        assert_eq!(out.hits.len(), 1);

        let req = server.await.unwrap();
        assert!(req.starts_with("POST /collections/support_transcripts/points/search HTTP/1.1"));
    }
}
