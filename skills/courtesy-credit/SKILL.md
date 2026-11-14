---
name: courtesy-credit
description: Evaluates delay criteria against cx_policy_knowledge SOPs and formats a Human-in-the-Loop (HITL) courtesy credit proposal card up to $20.00 USD.
allowed_tools:
  - policy_rag
  - api_get_order_details
  - api_get_tracking
  - api_submit_credit_command
endpoints:
  - /collections/cx_policy_knowledge/points/search
  - POST /api/v1/commands/orders/issue-credit (HITL UI approval only; never called directly by agent)
---

# Courtesy Credit Runbook

Use this skill when evaluating whether a delayed or impacted order qualifies for a courtesy credit.

## Workflow
1. **Verify Order & Delay:** Ensure order details (`api_get_order_details`) and tracking milestones (`api_get_tracking`) confirm the delay or service failure.
2. **Verify SOP Policy:** Query `policy_rag` against `cx_policy_knowledge` to confirm courtesy credit eligibility rules and tier limits.
3. **Enforce $20.00 Ceiling:** The courtesy credit amount MUST be strictly `> $0.00` and `<= $20.00` USD (`2000` cents). Requests exceeding `$20.00` must be rejected or escalated for supervisor review.
4. **Emit HITL Proposal Card:** Call `api_submit_credit_command` to generate a `HitlProposalCard` (`requires_human_approval: true`, `status: PendingApproval`).

## Structural HITL Safety
- **NEVER** execute `POST /api/v1/commands/orders/issue-credit` directly from the agent.
- The human specialist must explicitly click **Approve** in the CX Copilot UI before `ferrous-api` executes the financial mutation.
