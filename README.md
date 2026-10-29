# ferrous-agent

[← Back to FerrousCX Root Architecture](https://github.com/reidlai/ferrous-cx)

[![Component Type: Agentic AI Core](https://img.shields.io/badge/component-Agentic%20AI%20Core-blue.svg)](https://github.com/reidlai/ferrous-cx)
[![Framework: rig-core](https://img.shields.io/badge/framework-rig--core%20(Pure%20Rust)-orange.svg)](https://github.com/0xPlaygrounds/rig)
[![Provider: Pluggable Multi-Model](https://img.shields.io/badge/LLMs-Multi--Provider%20(Ollama%20%7C%20HF%20%7C%20Claude%20%7C%20OpenAI)-purple.svg)](#-pluggable-multi-provider-llm-engine)
[![Gatekeeper: context-sanitizer](https://img.shields.io/badge/security-context--sanitizer%20(DLP%20%2B%20Ingress)-brightgreen.svg)](#-context-sanitizer-gatekeeper)
[![API Client: ferrous-api](https://img.shields.io/badge/integration-ferrous--api%20Client-blue.svg)](#-modular-skills--ferrous-api-tooling-suite)
[![DevSecOps CI: Tier 1](https://img.shields.io/badge/DevSecOps%20CI-Tier%201%20Passing-brightgreen.svg)](.github/workflows/devsecops-ci.yml)
[![Pre-Commit Enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)

**`ferrous-agent`** is the compiled, pure-Rust conversational AI and agentic orchestration submodule for the **FerrousCX** enterprise platform. Powered natively by **`rig` (`rig-core`)**, `ferrous-agent` drives the **CX Copilot** workbench, orchestrating real-time reasoning, tool invocation, and decision-support proposal cards for human customer service specialists (Carlos, Elena).

Rather than binding to a single AI vendor or local model, `ferrous-agent` provides a **universal, pluggable multi-provider LLM abstraction**. It supports local open-weights (Ollama, vLLM), serverless cloud endpoints (Hugging Face, OpenRouter, Groq), and frontier cloud APIs (Anthropic Claude, OpenAI, Google Gemini)—all configured dynamically via environment variables with zero code changes and zero cross-process Python IPC latency.

This submodule establishes:
1. **Pluggable Multi-Provider LLM Orchestration:** Pure-Rust driver utilizing `rig-core` provider clients with zero Python runtime dependencies and zero-cost in-process tool dispatch.
2. **In-Process `context-sanitizer` Security Gatekeeper:**  
   - **Ingress Firewall:** Blocks direct and indirect prompt injection, jailbreaks, and system prompt exfiltration across all models.
   - **Egress DLP:** Redacts and pseudonymizes sensitive PII (Brazilian CPF, credit cards, emails, phone numbers) before prompts leave the local process for external or local LLM inference.
3. **Modular Skills & `ferrous-api` Invocation Tools:**  
   Houses specialized agent skills (`order-investigation`, `courtesy-credit`, `carrier-escalation`, `replay-control`) and strongly typed `rig::tool::Tool` client implementations that directly call the REST, query, and command endpoints defined in **`ferrous-api`**.
4. **Structural Human-in-the-Loop (HITL) Safety:**  
   The agent is strictly non-autonomous regarding financial or state mutations. It produces structured proposal cards (`HitlProposalCard`) that require physical human click approval in the UI, which then triggers the authorized mutation in `ferrous-api`.
5. **Native In-Process LLM Benchmark Suite (`src/benchmark/`):**  
   An asynchronous evaluation harness measuring microsecond Time-To-First-Token (TTFT), Inter-Token Latency (ITL) jitter, and TPS via `tokio::time::Instant`, streaming live telemetry over Axum SSE, and persisting Arrow `RecordBatch` runs to Delta Lake (`delta/llm_benchmarks/`) for sub-3ms DataFusion SQL comparison queries.
6. **OpenTelemetry GenAI Semantic Conventions:**  
   Natively instruments reasoning steps, token metrics, and tool execution with standard `gen_ai.*` attributes adhering to universal distributed tracing (AD-23).

---

## 🤖 Pluggable Multi-Provider LLM Engine

`ferrous-agent` abstracts LLM interactions behind a provider-agnostic builder, allowing seamless switching between local open-weights and cloud APIs:

### 1. Supported Provider Tiers

| Provider Tier | Target Models / Backends | Transport & Auth | Intended Role |
| :--- | :--- | :--- | :--- |
| **Local Open-Weights** | Ollama, vLLM (`Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct`) | HTTP REST (`http://localhost:11434` / `v1`) | Zero external cloud spend; air-gapped demo |
| **Serverless Cloud** | Hugging Face Inference API (`Qwen/Qwen2.5-7B-Instruct`), OpenRouter, Groq | HTTPS / `HF_TOKEN` Bearer | Lightweight zero-daemon default |
| **Frontier Cloud APIs**| Anthropic Claude (`claude-3-5-sonnet`), OpenAI (`gpt-4o`), Google Gemini (`gemini-1.5-pro`) | HTTPS / `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` | Complex multi-turn dispute reasoning & benchmark baselines |

### 2. Dynamic Runtime Configuration

Provider and model selection are configured dynamically at runtime without recompilation:

```bash
# Option A: Serverless Cloud via Hugging Face (Default)
export LLM_PROVIDER="huggingface"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Option B: Local Open-Weights via Ollama
export LLM_PROVIDER="ollama"
export LLM_MODEL="qwen2.5:7b"
export LLM_API_BASE="http://localhost:11434"

# Option C: Local High-Throughput via vLLM
export LLM_PROVIDER="openai" # vLLM exposes OpenAI-compatible /v1 endpoints
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

# Option D: Frontier Cloud via Anthropic Claude
export LLM_PROVIDER="anthropic"
export LLM_MODEL="claude-3-5-sonnet-20241022"
export ANTHROPIC_API_KEY="sk-ant-your_key_here"

# Option E: Frontier Cloud via OpenAI
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4o"
export OPENAI_API_KEY="sk-proj-your_key_here"

# Option F: Frontier Cloud via Google Gemini
export LLM_PROVIDER="gemini"
export LLM_MODEL="gemini-1.5-pro"
export GEMINI_API_KEY="AIzaSy_your_key_here"
```

### 3. Provider-Agnostic Agent Construction

Regardless of the active provider, all agent skills and tools implement the identical `rig::tool::Tool` trait, guaranteeing zero-IPC in-process tool dispatch:

```rust
use rig::providers::{anthropic, huggingface, openai};
use rig::agent::Agent;

pub enum LlmClient {
    HuggingFace(huggingface::Client),
    Anthropic(anthropic::Client),
    OpenAi(openai::Client),
}

pub fn create_agent(config: &AgentConfig) -> Agent<impl rig::completion::CompletionModel> {
    match config.provider.as_str() {
        "anthropic" => {
            let client = anthropic::Client::from_env();
            client.agent(&config.model)
                .tool(ApiGetOrderTool::new(&config.api_url, &config.m2m_token))
                .tool(ApiSubmitCreditCommandTool::new(&config.api_url, &config.m2m_token))
                .preamble(&config.system_prompt)
                .build()
        },
        "openai" | "ollama" => {
            let client = openai::Client::from_url(&config.api_key, &config.api_base);
            client.agent(&config.model)
                .tool(ApiGetOrderTool::new(&config.api_url, &config.m2m_token))
                .tool(ApiSubmitCreditCommandTool::new(&config.api_url, &config.m2m_token))
                .preamble(&config.system_prompt)
                .build()
        },
        _ => {
            // Default: Hugging Face serverless
            let client = huggingface::Client::from_env();
            client.agent(&config.model)
                .tool(ApiGetOrderTool::new(&config.api_url, &config.m2m_token))
                .tool(ApiSubmitCreditCommandTool::new(&config.api_url, &config.m2m_token))
                .preamble(&config.system_prompt)
                .build()
        }
    }
}
```

---

## 🛡️ `context-sanitizer` Gatekeeper

`context-sanitizer` is an in-process Rust module that acts as a bidirectional firewall around the LLM reasoning loop:

```mermaid
flowchart TD
    subgraph Ingress ["1. Ingress Security (Anti-Hacking)"]
        RawPrompt["Inbound User Prompt / Ticket Context"]
        IngressFilter["context-sanitizer Ingress Firewall<br/>• Prompt Injection Detection<br/>• Jailbreak & Heuristic Filter<br/>• System Prompt Integrity Guard"]
    end

    subgraph Egress_DLP ["2. Egress Data Loss Prevention (PII Masking)"]
        DLPFilter["context-sanitizer Egress DLP<br/>• Brazilian CPF Masking (***.***.789-**)<br/>• Credit Card PAN Redaction (****-1234)<br/>• Email / Phone Pseudonymization"]
    end

    subgraph Rig_Core ["3. rig-core Reasoning Engine"]
        Agent["CX Copilot (rig-core)<br/>(Pluggable Multi-Provider LLM)"]
        Skills["Modular Skills Engine<br/>• order-investigation<br/>• courtesy-credit<br/>• carrier-escalation"]
        Tools["Strongly-Typed Tools Suite<br/>• ferrous-api REST Client Tools<br/>• DataFusion Query Tool<br/>• Qdrant SOP Policy RAG Tool"]
    end

    subgraph Output ["4. Safe Decision Output"]
        HITLCard["Human-in-the-Loop Proposal Card<br/>[Approve $15 Courtesy Credit]"]
    end

    RawPrompt --> IngressFilter
    IngressFilter --> DLPFilter
    DLPFilter -->|Sanitized Tokens (Zero Raw PII)| Agent
    Agent <--> Skills
    Skills <--> Tools
    Agent --> HITLCard
```

---

## 🛠️ Modular Skills & `ferrous-api` Tooling Suite

`ferrous-agent` organizes its capabilities into declarative **Skills** and compiled **`rig::tool::Tool`** implementations that call the endpoints exposed by **`ferrous-api`**:

### 1. Modular Skills Catalog (`skills/`)

| Skill Directory | Target `ferrous-api` Endpoints | Role & Execution Workflow |
| :--- | :--- | :--- |
| **`skills/order-investigation/`** | `GET /api/v1/orders/{order_id}`<br/>`GET /api/v1/orders/{order_id}/tracking` | Multi-step investigation diagnosing order status, shipment milestones, carrier delay duration, and customer history. |
| **`skills/courtesy-credit/`** | `POST /api/v1/commands/orders/issue-credit` | Evaluates delay criteria against `cx_policy_knowledge` SOPs, checks Carlos/Elena's credit limit in Kanidm, and formats `HitlProposalCard`. |
| **`skills/carrier-escalation/`** | `POST /api/v1/commands/disputes/escalate` | Detects carrier SLA breach patterns (e.g. Correios SEDEX > 4 days) and files formal carrier dispute tickets. |
| **`skills/replay-control/`** | `POST /api/v1/replay/start`<br/>`POST /api/v1/replay/pause` | Controls simulation speed (1x, 10x, 50x) and replayer profiles via `ferrous-api` control routes. |

### 2. `rig::tool::Tool` Implementation Calling `ferrous-api`

Tools wrap strongly typed HTTP requests against `ferrous-api`, authenticating via Kanidm M2M Bearer tokens:

```rust
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct GetOrderArgs {
    pub order_id: String,
}

#[derive(Serialize, Deserialize)]
pub struct OrderDetails {
    pub order_id: String,
    pub status: String,
    pub days_delayed: i64,
    pub freight_cents: i64,
}

pub struct ApiGetOrderTool {
    pub api_base_url: String,
    pub auth_token: String,
}

impl Tool for ApiGetOrderTool {
    const NAME: &'static str = "api_get_order_details";
    type Error = ToolError;
    type Args = GetOrderArgs;
    type Output = OrderDetails;

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = format!("{}/api/v1/orders/{}", self.api_base_url, args.order_id);
        let client = reqwest::Client::new();
        
        let response = client
            .get(&url)
            .bearer_auth(&self.auth_token)
            .send()
            .await?
            .json::<OrderDetails>()
            .await?;

        Ok(response)
    }
}
```

---

## ⚡ Native In-Process LLM Benchmark Suite (`src/benchmark/`)

In compliance with **AD-22 (FR-28 / NFR-14)**, `ferrous-agent` embeds an in-process, asynchronous benchmark suite replacing heavyweight Python evaluation frameworks (Promptfoo, DeepEval).

### Benchmark Architecture & Metrics
1. **Microsecond Latency Instrumentation:** Measures Time-To-First-Token (TTFT), Inter-Token Latency (ITL) jitter, and Tokens Per Second (TPS) via `tokio::time::Instant` and `rig-core` `stream_prompt()`.
2. **Contract-First Tool Calling Validation:** Validates arguments against production Rust tool traits (`ApiGetOrderTool`, `ApiSubmitCreditCommandTool`) using `schemars` and pure-Rust `jsonschema` in sub-millisecond time.
3. **Domain Guardrail Assertions:** Validates business invariants (AD-17 $20 credit limit ceiling) and asserts PII masking via `context-sanitizer`.
4. **SSE Telemetry Streaming:** Broadcasts real-time chunk streams over `GET /api/v1/benchmarks/:run_id/stream` to the Svelte 5 Copilot Benchmarking Studio.
5. **Delta Lake & Arrow Lakehouse Persistence:** Batches benchmark records as Apache Arrow `RecordBatch` tables and appends them atomically to `delta/llm_benchmarks/` with ACID transaction logs via `delta-lake-rs`.
6. **Sub-3ms DataFusion SQL Comparisons:** Enables immediate analytical SQL queries across historical benchmark runs:
   ```sql
   SELECT model_id, provider,
          ROUND(AVG(ttft_ms), 1) AS avg_ttft_ms,
          ROUND(AVG(tps), 1) AS avg_tps,
          ROUND(AVG(tool_call_accuracy) * 100, 1) AS tool_acc_pct
   FROM llm_benchmarks
   GROUP BY model_id, provider
   ORDER BY avg_ttft_ms ASC;
   ```

---

## 📁 Repository Layout

```text
ferrous-agent/
├── .github/workflows/
│   └── devsecops-ci.yml        # Tier 1 Rust CI (clippy, cargo-audit, test)
├── .pre-commit-config.yaml      # Pre-commit hooks (cargo fmt, gitleaks)
├── Cargo.toml                   # rig-core, reqwest, qdrant-client, arrow, delta-lake-rs
├── skills/                      # Declarative Agent Skills & Runbooks
│   ├── order-investigation/     # Multi-step order diagnosis workflow
│   ├── courtesy-credit/         # Credit proposal & policy verification
│   ├── carrier-escalation/      # Carrier SLA breach dispute workflow
│   └── replay-control/          # Simulation control orchestration
├── scripts/
│   ├── test-api-tools.sh        # Standalone verification script calling ferrous-api
│   └── seed-agent-test.py       # Mock environment test harness
├── src/
│   ├── lib.rs                   # Library crate entrypoint
│   ├── config.rs                # Dynamic LLM provider & environment configuration
│   ├── sanitizer/               # context-sanitizer Gatekeeper Module
│   │   ├── mod.rs
│   │   ├── ingress.rs           # Anti-injection & jailbreak detection
│   │   ├── pii_rules.rs         # CPF, credit card, phone, email regex & algorithms
│   │   └── de_anonymizer.rs     # Session-scoped reversible token mapper
│   ├── agent/                   # rig-core Agent Pipelines
│   │   ├── mod.rs
│   │   ├── provider.rs          # Multi-provider factory (Ollama, HF, Anthropic, OpenAI)
│   │   ├── copilot.rs           # CX Copilot agent definition & prompt builder
│   │   └── streaming.rs         # Token streaming & abort handler
│   ├── benchmark/               # Native In-Process LLM Benchmark Harness (AD-22)
│   │   ├── mod.rs               # Benchmark runner & suite registry
│   │   ├── runner.rs            # Async tokio execution & microsecond timing
│   │   ├── metrics.rs           # TTFT, ITL jitter, TPS, tool-call accuracy
│   │   ├── cases.rs             # Golden evaluation test cases & guardrail assertions
│   │   └── storage.rs           # Arrow RecordBatch generation & Delta Lake persistence
│   ├── tools/                   # Strongly Typed rig::tool::Tool implementations
│   │   ├── mod.rs
│   │   ├── api_order.rs         # Invokes ferrous-api GET /api/v1/orders/{id}
│   │   ├── api_tracking.rs      # Invokes ferrous-api GET /api/v1/orders/{id}/tracking
│   │   ├── api_credit.rs        # Validates & prepares POST /api/v1/commands/...
│   │   ├── policy_rag.rs        # Qdrant cx_policy_knowledge RAG tool
│   │   └── transcripts_rag.rs   # Qdrant support_transcripts few-shot tool
│   ├── client/                  # Generated ferrous-api client wrapper
│   │   ├── mod.rs
│   │   └── http_client.rs       # Authenticated reqwest client (Kanidm M2M Bearer)
│   ├── models/                  # HitlProposalCard and JSON schemas
│   └── main.rs                  # Standalone agent daemon & benchmark runner CLI
└── README.md                    # Submodule specification & quickstart
```

---

## 🛠️ Developer Quickstart & Verification

### 1. Build and Run Unit Tests
```bash
cargo build --release
cargo test --all-targets --verbose
```

### 2. Configure Active LLM Provider
```bash
# Select provider (huggingface | ollama | anthropic | openai | gemini)
export LLM_PROVIDER="huggingface"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_your_token_here"
```

### 3. Verify `ferrous-api` Tools Standalone
```bash
# Set ferrous-api base URL and auth token (Kanidm M2M)
export FERROUS_API_URL="http://localhost:8080"
export FERROUS_AUTH_TOKEN="Bearer <token>"

bash scripts/test-api-tools.sh
```

### 4. Run In-Process LLM Benchmark Suite
```bash
# Execute benchmark suite against active provider
cargo run --bin ferrous-agent -- benchmark --suite all --output delta
```

---

## 🛡️ DevSecOps & Quality Assurance (Tier 1 Rust)

- **API Contract Synchronization:** CI validates tool request/response schemas against `ferrous-api` OpenAPI v3 outputs (`gen/openapi/openapi3.json`).
- **Multi-Model Provider Abstraction:** Validates that tool traits and agent pipelines compile and dispatch uniformly across local and cloud backends.
- **Injection Benchmark Suite:** Automated tests challenging the ingress gatekeeper with OWASP Top 10 LLM prompt injection and jailbreak payloads.
- **PII Leakage Verification:** CI asserts zero CPF, credit card, or email patterns pass unmasked through `context-sanitizer`.
- **Zero-Warning Linting:** `cargo clippy --all-targets -- -D warnings`.
- **Sub-Millisecond Overhead:** `context-sanitizer` executes in **< 1.0 ms (p99)** on standard CPU.
