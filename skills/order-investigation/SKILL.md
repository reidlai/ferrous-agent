---
name: order-investigation
description: Multi-step investigation diagnosing order status, shipment milestones, carrier delay duration, and customer history.
allowed_tools:
  - api_get_order_details
  - api_get_tracking
  - transcripts_rag
  - policy_rag
endpoints:
  - GET /api/v1/orders/{order_id}
  - GET /api/v1/orders/{order_id}/tracking
---

# Order Investigation Runbook

Use this skill when a customer specialist (e.g., Carlos or Elena) asks about an order's current status, delayed delivery, missing package, or shipment timeline.

## Workflow
1. **Fetch Order Details:** Call `api_get_order_details` with the customer's `order_id` (`GET /api/v1/orders/{order_id}`).
2. **Fetch Shipment Tracking:** Call `api_get_tracking` with `order_id` (`GET /api/v1/orders/{order_id}/tracking`) to inspect carrier milestones, last scan timestamp, and `days_delayed`.
3. **Check Prior Context (Optional):** If historical context or similar dispute resolutions are needed, query `transcripts_rag` (`support_transcripts`) or `policy_rag` (`cx_policy_knowledge`).
4. **Synthesize Findings:** Summarize order status, carrier, delay duration, and next recommended action clearly for the human operator.

## Constraints
- Read-only investigation only. Never attempt to mutate order state or control stream replay.
- Never expose raw PII (CPF, credit card PAN, email, phone) in outbound prompts.
