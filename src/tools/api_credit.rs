use rig_core::tool::{args_schema, Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::models::{HitlProposalCard, MAX_COURTESY_CREDIT_USD};
use crate::tools::ToolError;

/// Arguments for [`ApiSubmitCreditCommandTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SubmitCreditArgs {
    /// Target order identifier.
    pub order_id: String,
    /// Courtesy credit amount in USD (must be > 0.0 and <= 20.00).
    pub amount_usd: f64,
    /// Business justification for the courtesy credit.
    pub reason: String,
    /// Optional SOP / policy reference from `policy_rag`.
    #[serde(default)]
    pub policy_reference: Option<String>,
}

/// Structural HITL tool that validates the `$20.00` courtesy credit ceiling and emits a
/// [`HitlProposalCard`] requiring explicit human operator click approval.
///
/// **Safety invariant:** This tool holds NO HTTP client and NEVER calls
/// `POST /api/v1/commands/orders/issue-credit`.
#[derive(Clone, Debug, Default)]
pub struct ApiSubmitCreditCommandTool;

impl ApiSubmitCreditCommandTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for ApiSubmitCreditCommandTool {
    const NAME: &'static str = "api_submit_credit_command";
    type Args = SubmitCreditArgs;
    type Output = HitlProposalCard;
    type Error = ToolError;

    fn description(&self) -> String {
        format!(
            "Validate a courtesy credit request (<= ${MAX_COURTESY_CREDIT_USD:.2} USD) and emit a Human-in-the-Loop HitlProposalCard requiring operator approval. Never executes a direct mutation."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        HitlProposalCard::new_courtesy_credit(
            args.order_id,
            args.amount_usd,
            args.reason,
            args.policy_reference,
        )
        .map_err(ToolError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ProposalAction, ProposalStatus};

    #[tokio::test]
    async fn emits_hitl_proposal_card_within_ceiling_without_http() {
        let tool = ApiSubmitCreditCommandTool::new();
        let mut ctx = ToolContext::new();

        let card = tool
            .call(
                &mut ctx,
                SubmitCreditArgs {
                    order_id: "ORD-500".to_string(),
                    amount_usd: 15.0,
                    reason: "Carrier delayed 5 days".to_string(),
                    policy_reference: Some("SOP-DELAY-02".to_string()),
                },
            )
            .await
            .expect("$15 credit should emit HitlProposalCard");

        assert!(card.requires_human_approval);
        assert_eq!(card.status, ProposalStatus::PendingApproval);
        assert_eq!(card.target_endpoint, "/api/v1/commands/orders/issue-credit");
        assert!(matches!(
            card.action,
            ProposalAction::IssueCourtesyCredit {
                amount_cents: 1500,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn rejects_credit_above_twenty_dollars_and_non_positive_amounts() {
        let tool = ApiSubmitCreditCommandTool::new();
        let mut ctx = ToolContext::new();

        let over = tool
            .call(
                &mut ctx,
                SubmitCreditArgs {
                    order_id: "ORD-500".to_string(),
                    amount_usd: 25.0,
                    reason: "Over limit".to_string(),
                    policy_reference: None,
                },
            )
            .await
            .expect_err("$25 must be rejected");
        assert!(matches!(
            over,
            ToolError::CreditLimitExceeded {
                requested_usd: 25.0,
                max_usd: 20.0
            }
        ));

        for bad_amount in [0.0, -10.0, f64::NAN, f64::INFINITY] {
            let err = tool
                .call(
                    &mut ctx,
                    SubmitCreditArgs {
                        order_id: "ORD-500".to_string(),
                        amount_usd: bad_amount,
                        reason: "Invalid".to_string(),
                        policy_reference: None,
                    },
                )
                .await
                .expect_err("non-positive or non-finite amount must be rejected");
            assert!(matches!(err, ToolError::InvalidAmount(_)));
        }
    }
}
