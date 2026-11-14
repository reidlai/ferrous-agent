//! Strongly typed domain models, HITL proposal cards, and structured DataFusion extractors.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use rig_core::completion::CompletionRequest;
use rig_core::driver::DynModel;
use rig_core::operation::Completion;
use rig_core::structured_output::{missing_required_fields, prompted_augmentation};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::agent::provider::LlmClient;
use crate::agent::AgentError;

/// Maximum autonomous proposal ceiling for courtesy credits in USD.
pub const MAX_COURTESY_CREDIT_USD: f64 = 20.0;

/// Maximum autonomous proposal ceiling for courtesy credits in integer cents.
pub const MAX_COURTESY_CREDIT_CENTS: i64 = 2000;

/// Approval lifecycle state of a [`HitlProposalCard`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    PendingApproval,
    Approved,
    Rejected,
    Expired,
}

/// Domain action proposed by the agent for explicit human operator approval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProposalAction {
    /// Propose issuing a courtesy credit up to `$20.00` USD (`2000` cents).
    IssueCourtesyCredit {
        order_id: String,
        amount_usd: f64,
        amount_cents: i64,
        reason: String,
    },
    /// Propose escalating a carrier SLA breach dispute.
    EscalateCarrierDispute {
        order_id: String,
        carrier: String,
        reason: String,
    },
}

/// Errors produced when validating a [`HitlProposalCard`] invariant.
#[derive(Debug, Error, PartialEq)]
pub enum ProposalValidationError {
    #[error("order_id must not be empty")]
    EmptyOrderId,
    #[error("reason must not be empty")]
    EmptyReason,
    #[error("courtesy credit amount must be finite and greater than $0.00 (got {0})")]
    InvalidAmount(f64),
    #[error(
        "courtesy credit ${requested_usd:.2} exceeds the ${max_usd:.2} autonomous proposal ceiling"
    )]
    CreditLimitExceeded { requested_usd: f64, max_usd: f64 },
}

/// Structural Human-in-the-Loop (HITL) proposal card emitted instead of direct mutations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HitlProposalCard {
    /// Time-ordered UUIDv7 identifier for UI tracking and audit logs.
    #[schemars(with = "String")]
    pub proposal_id: Uuid,
    /// Structured domain action to execute only after operator click approval.
    pub action: ProposalAction,
    /// Concise human-readable summary rendered on the UI card.
    pub summary: String,
    /// Always `true`: the agent never executes financial or state mutations autonomously.
    pub requires_human_approval: bool,
    /// Initialized to [`ProposalStatus::PendingApproval`].
    pub status: ProposalStatus,
    /// Target `ferrous-api` command endpoint that the UI invokes upon human approval.
    pub target_endpoint: String,
    /// Optional SOP / policy reference supporting the proposal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_reference: Option<String>,
}

impl HitlProposalCard {
    /// Validate invariants and construct a [`HitlProposalCard`] for a courtesy credit.
    ///
    /// Enforces `0.0 < amount_usd <= 20.0` (`<= 2000` cents) and always sets
    /// `requires_human_approval = true` and `status = ProposalStatus::PendingApproval`.
    pub fn new_courtesy_credit(
        order_id: impl Into<String>,
        amount_usd: f64,
        reason: impl Into<String>,
        policy_reference: Option<String>,
    ) -> Result<Self, ProposalValidationError> {
        let order_id = order_id.into().trim().to_string();
        if order_id.is_empty() {
            return Err(ProposalValidationError::EmptyOrderId);
        }
        let reason = reason.into().trim().to_string();
        if reason.is_empty() {
            return Err(ProposalValidationError::EmptyReason);
        }
        if !amount_usd.is_finite() || amount_usd <= 0.0 {
            return Err(ProposalValidationError::InvalidAmount(amount_usd));
        }

        let amount_cents = (amount_usd * 100.0).round() as i64;
        if amount_usd > MAX_COURTESY_CREDIT_USD || amount_cents > MAX_COURTESY_CREDIT_CENTS {
            return Err(ProposalValidationError::CreditLimitExceeded {
                requested_usd: amount_usd,
                max_usd: MAX_COURTESY_CREDIT_USD,
            });
        }
        if amount_cents <= 0 {
            return Err(ProposalValidationError::InvalidAmount(amount_usd));
        }
        let normalized_usd = (amount_cents as f64) / 100.0;

        let summary = format!(
            "Approve ${:.2} USD courtesy credit for order {} ({})",
            normalized_usd, order_id, reason
        );

        Ok(Self {
            proposal_id: Uuid::now_v7(),
            action: ProposalAction::IssueCourtesyCredit {
                order_id,
                amount_usd: normalized_usd,
                amount_cents,
                reason,
            },
            summary,
            requires_human_approval: true,
            status: ProposalStatus::PendingApproval,
            target_endpoint: "/api/v1/commands/orders/issue-credit".to_string(),
            policy_reference: policy_reference
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty()),
        })
    }
}

/// Line item within an order returned by `GET /api/v1/orders/{order_id}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrderItem {
    #[serde(default)]
    pub sku: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub quantity: i64,
    #[serde(default)]
    pub unit_price_cents: i64,
}

/// Order details returned by `ferrous-api` (`GET /api/v1/orders/{order_id}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrderDetails {
    pub order_id: String,
    #[serde(default)]
    pub customer_id: String,
    pub status: String,
    #[serde(default)]
    pub total_amount_cents: i64,
    #[serde(default)]
    pub currency: String,
    #[serde(default)]
    pub days_delayed: Option<i64>,
    #[serde(default)]
    pub freight_cents: Option<i64>,
    #[serde(default)]
    pub carrier: Option<String>,
    #[serde(default)]
    pub items: Vec<OrderItem>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// Individual shipment tracking milestone returned by `GET /api/v1/orders/{order_id}/tracking`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrackingMilestone {
    pub timestamp: String,
    pub status: String,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Shipment tracking summary returned by `GET /api/v1/orders/{order_id}/tracking`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TrackingDetails {
    pub order_id: String,
    #[serde(default)]
    pub carrier: String,
    #[serde(default)]
    pub tracking_code: Option<String>,
    pub current_status: String,
    #[serde(default)]
    pub days_delayed: i64,
    #[serde(default)]
    pub sla_breached: bool,
    #[serde(default)]
    pub estimated_delivery: Option<String>,
    #[serde(default)]
    pub milestones: Vec<TrackingMilestone>,
}

/// Single vector search hit returned by Qdrant `/collections/{collection}/points/search`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RagHit {
    pub id: serde_json::Value,
    pub score: f64,
    #[serde(default)]
    pub payload: BTreeMap<String, serde_json::Value>,
}

/// Qdrant search REST envelope (`{ "result": [ ... ] }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RagSearchResponse {
    #[serde(default)]
    pub result: Vec<RagHit>,
}

/// Aggregated carrier delay metrics extracted from tabular DataFusion SQL query results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CarrierDelayMetrics {
    /// Carrier identifier (e.g., `"Correios SEDEX"`).
    pub carrier: String,
    /// Total shipments evaluated in the DataFusion window.
    pub total_shipments: u64,
    /// Count of shipments exceeding the promised delivery SLA.
    pub delayed_shipments: u64,
    /// Mean delay duration in days across delayed shipments.
    pub avg_delay_days: f64,
    /// 95th percentile delay duration in days.
    pub p95_delay_days: f64,
    /// Fraction (`0.0..=1.0`) of shipments breaching SLA.
    pub sla_breach_rate: f64,
    /// Recommended operational or escalation action.
    pub recommended_action: String,
}

impl CarrierDelayMetrics {
    /// Deterministically parse a [`CarrierDelayMetrics`] from JSON or a single-row DataFusion JSON array.
    pub fn from_tabular_json(raw: &str) -> Result<Self, serde_json::Error> {
        parse_tabular_json(raw)
    }
}

/// Structured dispute profile for an individual delayed or contested order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrderDisputeProfile {
    pub order_id: String,
    pub carrier: String,
    pub days_delayed: i64,
    pub sla_breached: bool,
    pub eligible_for_courtesy_credit: bool,
    #[serde(default)]
    pub recommended_credit_usd: Option<f64>,
    pub summary: String,
}

impl OrderDisputeProfile {
    /// Deterministically parse an [`OrderDisputeProfile`] from JSON or a single-row DataFusion JSON array.
    pub fn from_tabular_json(raw: &str) -> Result<Self, serde_json::Error> {
        parse_tabular_json(raw)
    }
}

/// Parse either a JSON object or a single-row JSON array (`[{...}]`) emitted by DataFusion.
pub fn parse_tabular_json<T: DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    let value = parse_tabular_value(raw)?;
    serde_json::from_value(value)
}

fn parse_tabular_value(raw: &str) -> Result<serde_json::Value, serde_json::Error> {
    let cleaned = strip_markdown_fences(raw.trim());
    let value: serde_json::Value = serde_json::from_str(cleaned)?;
    match value {
        serde_json::Value::Array(mut rows) => {
            if rows.len() == 1 {
                Ok(rows.remove(0))
            } else {
                Err(serde::de::Error::custom(format!(
                    "expected a JSON object or single-row DataFusion array, got array of length {}",
                    rows.len()
                )))
            }
        }
        other => Ok(other),
    }
}

fn strip_markdown_fences(input: &str) -> &str {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let after_lang = rest.find('\n').map(|idx| &rest[idx + 1..]).unwrap_or(rest);
        if let Some(body) = after_lang.strip_suffix("```") {
            return body.trim();
        }
    }
    trimmed
}

/// Structured output extractor backed by `rig-core`'s [`DynModel<Completion>`] and `schemars`.
#[derive(Clone, Debug)]
pub struct StructuredExtractor<T> {
    model: DynModel<Completion>,
    preamble: Option<String>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> StructuredExtractor<T>
where
    T: DeserializeOwned + JsonSchema,
{
    /// Create a new [`StructuredExtractor`] around a `rig-core` completion model.
    pub fn new(model: DynModel<Completion>) -> Self {
        Self {
            model,
            preamble: None,
            _marker: PhantomData,
        }
    }

    /// Attach an optional domain preamble to the extraction request.
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        let value = preamble.into();
        if !value.trim().is_empty() {
            self.preamble = Some(value);
        }
        self
    }

    /// Return the JSON Schema generated by `schemars = "1.2"` for `T`.
    pub fn schema(&self) -> schemars::Schema {
        schemars::schema_for!(T)
    }

    /// Build the `rig-core` [`CompletionRequest`] configured with `output_schema` and
    /// `rig_core::structured_output::prompted_augmentation`.
    pub fn build_request(&self, input: &str) -> Result<CompletionRequest, serde_json::Error> {
        let schema = self.schema();
        let schema_str = serde_json::to_string_pretty(&schema)?;
        let augmentation = prompted_augmentation(&schema_str);
        let full_preamble = match &self.preamble {
            Some(base) => format!("{base}\n\n{augmentation}"),
            None => augmentation,
        };

        Ok(CompletionRequest::new(input)
            .preamble(full_preamble)
            .output_schema(schema))
    }

    /// Parse and validate a model completion text against `T`'s schema.
    pub fn parse_response(&self, response_text: &str) -> Result<T, AgentError> {
        let schema_val = serde_json::to_value(self.schema())
            .map_err(|e| AgentError::Stream(format!("schema serialization failed: {e}")))?;
        let value = parse_tabular_value(response_text)
            .map_err(|e| AgentError::Stream(format!("structured extraction parse failed: {e}")))?;
        let missing = missing_required_fields(&schema_val, &value);
        if !missing.is_empty() {
            return Err(AgentError::Stream(format!(
                "structured extraction missing required field(s): {}",
                missing.join(", ")
            )));
        }
        serde_json::from_value(value)
            .map_err(|e| AgentError::Stream(format!("structured extraction parse failed: {e}")))
    }

    /// Execute the extraction against the configured [`DynModel<Completion>`].
    pub async fn extract(&self, input: &str) -> Result<T, AgentError> {
        let request = self
            .build_request(input)
            .map_err(|e| AgentError::Stream(format!("schema serialization failed: {e}")))?;
        let response = self.model.call(request).await?;
        self.parse_response(&response.text())
    }
}

/// Build a [`StructuredExtractor<CarrierDelayMetrics>`] from an [`LlmClient`].
pub fn build_carrier_delay_extractor(
    client: &LlmClient,
) -> StructuredExtractor<CarrierDelayMetrics> {
    StructuredExtractor::new(client.rig_model().clone()).with_preamble(
        "Extract DataFusion SQL carrier delay metrics accurately into CarrierDelayMetrics JSON.",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::config::AgentConfig;

    #[test]
    fn hitl_proposal_card_enforces_twenty_dollar_ceiling() {
        let ok_card = HitlProposalCard::new_courtesy_credit(
            "ORD-1001",
            15.0,
            "Correios SEDEX delayed 5 days",
            Some("SOP-CC-01".to_string()),
        )
        .expect("$15 credit should succeed");

        assert!(ok_card.requires_human_approval);
        assert_eq!(ok_card.status, ProposalStatus::PendingApproval);
        assert_eq!(
            ok_card.target_endpoint,
            "/api/v1/commands/orders/issue-credit"
        );
        assert_eq!(
            ok_card.action,
            ProposalAction::IssueCourtesyCredit {
                order_id: "ORD-1001".to_string(),
                amount_usd: 15.0,
                amount_cents: 1500,
                reason: "Correios SEDEX delayed 5 days".to_string(),
            }
        );

        let boundary = HitlProposalCard::new_courtesy_credit("ORD-1001", 20.0, "Max allowed", None)
            .expect("$20.00 boundary should succeed");
        assert!(boundary.requires_human_approval);

        let over = HitlProposalCard::new_courtesy_credit("ORD-1001", 20.01, "Too high", None)
            .expect_err("$20.01 must be rejected");
        assert!(matches!(
            over,
            ProposalValidationError::CreditLimitExceeded { .. }
        ));

        let zero = HitlProposalCard::new_courtesy_credit("ORD-1001", 0.0, "Zero", None)
            .expect_err("$0.00 must be rejected");
        assert!(matches!(zero, ProposalValidationError::InvalidAmount(_)));

        let negative = HitlProposalCard::new_courtesy_credit("ORD-1001", -5.0, "Negative", None)
            .expect_err("negative amount must be rejected");
        assert!(matches!(
            negative,
            ProposalValidationError::InvalidAmount(_)
        ));
    }

    #[test]
    fn carrier_delay_metrics_parses_object_and_datafusion_row_array() {
        let single_row_array = r#"[{
            "carrier": "Correios SEDEX",
            "total_shipments": 120,
            "delayed_shipments": 18,
            "avg_delay_days": 4.5,
            "p95_delay_days": 7.2,
            "sla_breach_rate": 0.15,
            "recommended_action": "Escalate carrier SLA breach"
        }]"#;

        let parsed = CarrierDelayMetrics::from_tabular_json(single_row_array)
            .expect("single-row DataFusion array should parse");
        assert_eq!(parsed.carrier, "Correios SEDEX");
        assert_eq!(parsed.total_shipments, 120);
        assert_eq!(parsed.delayed_shipments, 18);
        assert!((parsed.avg_delay_days - 4.5).abs() < f64::EPSILON);

        let malformed = CarrierDelayMetrics::from_tabular_json(r#"{"carrier": "OnlyOneField"}"#);
        assert!(malformed.is_err());
    }

    #[test]
    fn structured_extractor_builds_schema_and_parses_fenced_json() {
        let env: HashMap<String, String> = [
            ("LLM_PROVIDER".to_string(), "ollama".to_string()),
            ("LLM_MODEL".to_string(), "qwen2.5:7b".to_string()),
        ]
        .into_iter()
        .collect();
        let cfg = AgentConfig::from_map(&env).expect("ollama config");
        let client = LlmClient::from_config(&cfg).expect("client");

        let extractor = build_carrier_delay_extractor(&client);
        let req = extractor
            .build_request("carrier=Correios SEDEX, total=50, delayed=10")
            .expect("request should build");
        assert!(req.output_schema.is_some());
        assert!(req
            .system_instructions()
            .unwrap_or_default()
            .contains("CarrierDelayMetrics"));

        let fenced = "```json\n{\"carrier\":\"Correios SEDEX\",\"total_shipments\":50,\"delayed_shipments\":10,\"avg_delay_days\":3.2,\"p95_delay_days\":5.0,\"sla_breach_rate\":0.2,\"recommended_action\":\"Monitor\"}\n```";
        let metrics = extractor
            .parse_response(fenced)
            .expect("fenced JSON should parse");
        assert_eq!(metrics.delayed_shipments, 10);

        let missing_err = extractor
            .parse_response(r#"{"carrier":"Correios SEDEX"}"#)
            .expect_err("missing required fields must fail");
        assert!(matches!(missing_err, AgentError::Stream(_)));
    }

    #[test]
    fn sub_cent_credit_is_normalized_and_order_dispute_profile_parses() {
        let card = HitlProposalCard::new_courtesy_credit("ORD-12", 15.126, "Delay", None)
            .expect("sub-cent input should normalize");
        assert_eq!(
            card.action,
            ProposalAction::IssueCourtesyCredit {
                order_id: "ORD-12".to_string(),
                amount_usd: 15.13,
                amount_cents: 1513,
                reason: "Delay".to_string(),
            }
        );

        assert!(CarrierDelayMetrics::from_tabular_json("[]").is_err());
        assert!(CarrierDelayMetrics::from_tabular_json("[{}, {}]").is_err());

        let profile_json = r#"[{
            "order_id": "ORD-900",
            "carrier": "Correios SEDEX",
            "days_delayed": 6,
            "sla_breached": true,
            "eligible_for_courtesy_credit": true,
            "recommended_credit_usd": 15.0,
            "summary": "SLA breached by 2 days"
        }]"#;
        let profile = OrderDisputeProfile::from_tabular_json(profile_json)
            .expect("OrderDisputeProfile should parse from DataFusion array");
        assert_eq!(profile.order_id, "ORD-900");
        assert!(profile.eligible_for_courtesy_credit);
        assert_eq!(profile.recommended_credit_usd, Some(15.0));
    }
}
