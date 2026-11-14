---
name: carrier-escalation
description: Detects carrier SLA breach patterns (e.g., Correios SEDEX > 4 days) and prepares formal carrier dispute escalation proposals.
allowed_tools:
  - api_get_order_details
  - api_get_tracking
  - policy_rag
  - transcripts_rag
endpoints:
  - GET /api/v1/orders/{order_id}
  - GET /api/v1/orders/{order_id}/tracking
  - POST /api/v1/commands/disputes/escalate (HITL UI approval only)
---

# Carrier Escalation Runbook

Use this skill when shipment tracking or DataFusion carrier delay metrics indicate a carrier SLA breach (for example, Correios SEDEX delayed > 4 days).

## Workflow
1. **Inspect Tracking & Delay Metrics:** Call `api_get_tracking` and `api_get_order_details` to confirm carrier name, service tier, and `days_delayed`.
2. **Validate SLA Breach Policy:** Query `policy_rag` (`cx_policy_knowledge`) for carrier-specific SLA thresholds and required evidence.
3. **Review Precedents:** Optionally query `transcripts_rag` (`support_transcripts`) for prior dispute escalation phrasing and outcomes.
4. **Prepare Escalation Summary:** Present the SLA breach evidence and recommended `EscalateCarrierDispute` proposal for human operator confirmation.

## Constraints
- Never bypass human operator approval for state-mutating dispute filings.
- Never issue or control stream replay operations.
