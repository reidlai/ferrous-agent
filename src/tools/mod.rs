//! Strongly typed `rig_core::tool::Tool` implementations for `ferrous-api` and Qdrant.

pub mod api_credit;
pub mod api_order;
pub mod api_tracking;
pub mod policy_rag;
pub mod transcripts_rag;

use std::sync::Arc;

use rig_core::completion::ToolDefinition;
use rig_core::tool::tool_definition;
use thiserror::Error;

use crate::client::{ClientError, FerrousApiClient};
use crate::models::ProposalValidationError;

pub use api_credit::{ApiSubmitCreditCommandTool, SubmitCreditArgs};
pub use api_order::{ApiGetOrderTool, GetOrderArgs};
pub use api_tracking::{ApiGetTrackingTool, GetTrackingArgs};
pub use policy_rag::{PolicyRagTool, RagQueryArgs, RagQueryOutput};
pub use transcripts_rag::TranscriptsRagTool;

/// Errors returned by `ferrous-agent` `rig_core::tool::Tool` implementations.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid tool input: {0}")]
    InvalidInput(String),

    #[error("invalid courtesy credit amount: {0}")]
    InvalidAmount(f64),

    #[error(
        "courtesy credit ${requested_usd:.2} exceeds the ${max_usd:.2} autonomous proposal ceiling"
    )]
    CreditLimitExceeded { requested_usd: f64, max_usd: f64 },

    #[error("API status {status} from {endpoint}: {body}")]
    ApiStatus {
        status: u16,
        endpoint: String,
        body: String,
    },

    #[error("HTTP client error: {0}")]
    Client(String),
}

impl From<ClientError> for ToolError {
    fn from(err: ClientError) -> Self {
        match err {
            ClientError::InvalidInput(msg) => Self::InvalidInput(msg),
            ClientError::ApiStatus {
                status,
                endpoint,
                body,
            } => Self::ApiStatus {
                status,
                endpoint,
                body,
            },
            other => Self::Client(other.to_string()),
        }
    }
}

impl From<ProposalValidationError> for ToolError {
    fn from(err: ProposalValidationError) -> Self {
        match err {
            ProposalValidationError::EmptyOrderId => {
                Self::InvalidInput("order_id must not be empty".to_string())
            }
            ProposalValidationError::EmptyReason => {
                Self::InvalidInput("reason must not be empty".to_string())
            }
            ProposalValidationError::InvalidAmount(amount) => Self::InvalidAmount(amount),
            ProposalValidationError::CreditLimitExceeded {
                requested_usd,
                max_usd,
            } => Self::CreditLimitExceeded {
                requested_usd,
                max_usd,
            },
        }
    }
}

/// Bundle of all five provider-agnostic `rig_core::tool::Tool` instances.
#[derive(Clone, Debug)]
pub struct FerrousToolSuite {
    pub get_order: ApiGetOrderTool,
    pub get_tracking: ApiGetTrackingTool,
    pub submit_credit: ApiSubmitCreditCommandTool,
    pub policy_rag: PolicyRagTool,
    pub transcripts_rag: TranscriptsRagTool,
}

impl FerrousToolSuite {
    /// Construct the full tool suite sharing a single authenticated [`FerrousApiClient`].
    pub fn new(client: Arc<FerrousApiClient>) -> Self {
        Self {
            get_order: ApiGetOrderTool::new(Arc::clone(&client)),
            get_tracking: ApiGetTrackingTool::new(Arc::clone(&client)),
            submit_credit: ApiSubmitCreditCommandTool::new(),
            policy_rag: PolicyRagTool::new(Arc::clone(&client)),
            transcripts_rag: TranscriptsRagTool::new(client),
        }
    }

    /// Return the provider-agnostic `rig_core::completion::ToolDefinition` list for all 5 tools.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            tool_definition(&self.get_order),
            tool_definition(&self.get_tracking),
            tool_definition(&self.submit_credit),
            tool_definition(&self.policy_rag),
            tool_definition(&self.transcripts_rag),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_suite_produces_identical_schemars_definitions() {
        let client = Arc::new(FerrousApiClient::new("http://localhost:8080", "tok"));
        let suite = FerrousToolSuite::new(client);
        let defs = suite.definitions();

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "api_get_order_details",
                "api_get_tracking",
                "api_submit_credit_command",
                "policy_rag",
                "transcripts_rag",
            ]
        );

        for def in &defs {
            assert!(!def.description.is_empty());
            assert_eq!(def.parameters["type"], "object");
            assert_eq!(def.parameters["additionalProperties"], false);
            assert!(
                !def.name.contains("replay"),
                "replay tools must never be registered"
            );
        }
    }
}
