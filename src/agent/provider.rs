use std::time::Instant;

use async_trait::async_trait;
use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::driver::DynModel;
use rig_core::operation::Completion;
use rig_core::providers::{anthropic, gemini, ollama, openai};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::agent::streaming::{adapt_rig_stream, BoxedChunkStream, StreamChunk};
use crate::agent::AgentError;
use crate::config::{AgentConfig, ProviderKind};
use crate::models::StructuredExtractor;
use crate::telemetry::{
    estimate_token_count, genai_chat_span, instrument_chunk_stream, record_completion_tokens,
};

/// Provider-agnostic completion and streaming dispatch interface.
#[async_trait]
pub trait CompletionDispatch: Send + Sync {
    /// Active provider tier.
    fn provider(&self) -> ProviderKind;

    /// Active model identifier.
    fn model_name(&self) -> &str;

    /// Execute a unary completion prompt, respecting `cancel`.
    async fn prompt(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String, AgentError>;

    /// Execute a unary completion prompt with registered tool definitions, respecting `cancel`.
    async fn prompt_with_tools(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        _tools: &[ToolDefinition],
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        self.prompt(system_preamble, prompt, cancel).await
    }

    /// Execute a streaming completion prompt, respecting `cancel`.
    async fn stream_prompt(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<BoxedChunkStream, AgentError>;
}

/// Multi-provider `rig-core` LLM client wrapping an erased `DynModel<Completion>`.
#[derive(Clone, Debug)]
pub struct LlmClient {
    provider: ProviderKind,
    model_name: String,
    model: DynModel<Completion>,
}

impl LlmClient {
    /// Construct the appropriate `rig-core` provider client from `AgentConfig`.
    pub fn from_config(config: &AgentConfig) -> Result<Self, AgentError> {
        let model = build_dyn_model(config)?;
        Ok(Self {
            provider: config.provider,
            model_name: config.model.clone(),
            model,
        })
    }

    /// Active provider kind.
    pub fn provider(&self) -> ProviderKind {
        self.provider
    }

    /// Configured model identifier.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Access the underlying erased `rig-core` completion model.
    pub fn rig_model(&self) -> &DynModel<Completion> {
        &self.model
    }

    /// Construct a [`StructuredExtractor<T>`] backed by this client's completion model.
    pub fn extractor<T>(&self) -> StructuredExtractor<T>
    where
        T: DeserializeOwned + JsonSchema,
    {
        StructuredExtractor::new(self.model.clone())
    }
}

#[async_trait]
impl CompletionDispatch for LlmClient {
    fn provider(&self) -> ProviderKind {
        self.provider
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn prompt(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        self.prompt_with_tools(system_preamble, prompt, &[], cancel)
            .await
    }

    async fn prompt_with_tools(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        tools: &[ToolDefinition],
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        if prompt.trim().is_empty() {
            return Err(AgentError::Stream("prompt must not be empty".to_string()));
        }
        if cancel.is_cancelled() {
            return Err(AgentError::Stream("prompt cancelled".to_string()));
        }

        let span = genai_chat_span(self.provider.as_str(), &self.model_name, "chat");
        let request = build_request_with_tools(system_preamble, prompt, tools);
        let fut = self.model.call(request).instrument(span.clone());

        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                span.record("otel.status_code", "ERROR");
                Err(AgentError::Stream("prompt cancelled".to_string()))
            }
            res = fut => {
                match res {
                    Ok(response) => {
                        let text = response.text();
                        record_completion_tokens(&span, estimate_token_count(&text));
                        Ok(text)
                    }
                    Err(err) => {
                        span.record("otel.status_code", "ERROR");
                        Err(err.into())
                    }
                }
            }
        }
    }

    async fn stream_prompt(
        &self,
        system_preamble: Option<&str>,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<BoxedChunkStream, AgentError> {
        if prompt.trim().is_empty() {
            return Err(AgentError::Stream("prompt must not be empty".to_string()));
        }
        if cancel.is_cancelled() {
            return Ok(Box::pin(futures::stream::once(async {
                Ok(StreamChunk::Cancelled)
            })));
        }

        let span = genai_chat_span(self.provider.as_str(), &self.model_name, "stream");
        let started_at = Instant::now();
        let request = build_request(system_preamble, prompt);
        let stream = {
            let _entered = span.enter();
            match self.model.stream(request) {
                Ok(s) => s,
                Err(err) => {
                    span.record("otel.status_code", "ERROR");
                    return Err(err.into());
                }
            }
        };
        let adapted = adapt_rig_stream(stream, cancel);
        Ok(instrument_chunk_stream(adapted, span, started_at))
    }
}

fn build_request(system_preamble: Option<&str>, prompt: &str) -> CompletionRequest {
    build_request_with_tools(system_preamble, prompt, &[])
}

fn build_request_with_tools(
    system_preamble: Option<&str>,
    prompt: &str,
    tools: &[ToolDefinition],
) -> CompletionRequest {
    let mut req = CompletionRequest::new(prompt);
    if let Some(preamble) = system_preamble.filter(|p| !p.trim().is_empty()) {
        req = req.preamble(preamble);
    }
    if !tools.is_empty() {
        req = req.tools(tools.iter().cloned());
    }
    req
}

fn require_key<'a>(config: &'a AgentConfig, env_var: &'static str) -> Result<&'a str, AgentError> {
    config
        .api_key_str()
        .ok_or_else(|| AgentError::MissingCredential {
            provider: config.provider.as_str().to_string(),
            env_var,
        })
}

fn build_openai_dialect(
    config: &AgentConfig,
    dialect: &openai::wire::Dialect,
    env_var: &'static str,
) -> Result<DynModel<Completion>, AgentError> {
    let key = require_key(config, env_var)?;
    let mut cfg = openai::OpenAIConfig::with_key(dialect, key);
    if let Some(base) = &config.api_base {
        cfg = cfg.with_base_url(base);
    }
    Ok(cfg.client().chat(&config.model).erase())
}

fn build_dyn_model(config: &AgentConfig) -> Result<DynModel<Completion>, AgentError> {
    match config.provider {
        ProviderKind::Ollama => {
            let mut cfg = ollama::OllamaConfig::new();
            if let Some(base) = &config.api_base {
                cfg = cfg.with_base_url(base);
            }
            if let Some(key) = config.api_key_str() {
                cfg = cfg.with_api_key(key);
            }
            Ok(cfg.client().completion(&config.model).erase())
        }
        ProviderKind::Vllm | ProviderKind::OpenAi => {
            build_openai_dialect(config, &openai::wire::OPENAI, "OPENAI_API_KEY")
        }
        ProviderKind::HuggingFace => {
            build_openai_dialect(config, &openai::wire::HUGGINGFACE, "HF_TOKEN")
        }
        ProviderKind::OpenRouter => {
            build_openai_dialect(config, &openai::wire::OPENROUTER, "OPENROUTER_API_KEY")
        }
        ProviderKind::Groq => build_openai_dialect(config, &openai::wire::GROQ, "GROQ_API_KEY"),
        ProviderKind::Anthropic => {
            let key = require_key(config, "ANTHROPIC_API_KEY")?;
            let mut cfg = anthropic::AnthropicConfig::new(key);
            if let Some(base) = &config.api_base {
                cfg = cfg.with_base_url(base);
            }
            Ok(cfg.client().completion(&config.model).erase())
        }
        ProviderKind::Gemini => {
            let key = require_key(config, "GEMINI_API_KEY")?;
            let mut cfg = gemini::GeminiConfig::new(key);
            if let Some(base) = &config.api_base {
                cfg = cfg.with_base_url(base);
            }
            Ok(cfg.client().completion(&config.model).erase())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn constructs_client_for_all_eight_providers() {
        let cases = [
            (
                ProviderKind::Ollama,
                vec![("LLM_PROVIDER", "ollama"), ("LLM_MODEL", "qwen2.5:7b")],
                "ollama",
            ),
            (
                ProviderKind::Vllm,
                vec![
                    ("LLM_PROVIDER", "vllm"),
                    ("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                    ("LLM_API_BASE", "http://localhost:8000/v1"),
                ],
                "openai",
            ),
            (
                ProviderKind::HuggingFace,
                vec![
                    ("LLM_PROVIDER", "huggingface"),
                    ("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                    ("HF_TOKEN", "hf_test_token"),
                ],
                "huggingface",
            ),
            (
                ProviderKind::OpenRouter,
                vec![
                    ("LLM_PROVIDER", "openrouter"),
                    ("LLM_MODEL", "openai/gpt-4o-mini"),
                    ("OPENROUTER_API_KEY", "or_test_token"),
                ],
                "openrouter",
            ),
            (
                ProviderKind::Groq,
                vec![
                    ("LLM_PROVIDER", "groq"),
                    ("LLM_MODEL", "llama-3.1-8b-instant"),
                    ("GROQ_API_KEY", "groq_test_token"),
                ],
                "groq",
            ),
            (
                ProviderKind::Anthropic,
                vec![
                    ("LLM_PROVIDER", "anthropic"),
                    ("LLM_MODEL", "claude-3-5-sonnet-20241022"),
                    ("ANTHROPIC_API_KEY", "sk-ant-test"),
                ],
                "anthropic",
            ),
            (
                ProviderKind::OpenAi,
                vec![
                    ("LLM_PROVIDER", "openai"),
                    ("LLM_MODEL", "gpt-4o"),
                    ("OPENAI_API_KEY", "sk-openai-test"),
                ],
                "openai",
            ),
            (
                ProviderKind::Gemini,
                vec![
                    ("LLM_PROVIDER", "gemini"),
                    ("LLM_MODEL", "gemini-1.5-pro"),
                    ("GEMINI_API_KEY", "AIza-test"),
                ],
                "gcp.gemini",
            ),
        ];

        for (expected_kind, pairs, expected_rig_name) in cases {
            let cfg = AgentConfig::from_map(&map(&pairs)).expect("config should parse");
            let client = LlmClient::from_config(&cfg).expect("LlmClient should construct");
            assert_eq!(client.provider(), expected_kind);
            assert_eq!(client.model_name(), cfg.model);
            assert_eq!(client.rig_model().name(), expected_rig_name);
            assert_eq!(client.rig_model().id(), Some(cfg.model.as_str()));
        }
    }

    #[test]
    fn build_request_handles_preamble_variants() {
        let with_preamble = build_request(Some("System rules"), "Hello");
        assert_eq!(with_preamble.system_instructions(), Some("System rules"));

        let without_preamble = build_request(None, "Hello");
        assert_eq!(without_preamble.system_instructions(), None);

        let whitespace_preamble = build_request(Some("  \t\n "), "Hello");
        assert_eq!(whitespace_preamble.system_instructions(), None);
    }

    #[tokio::test]
    async fn pre_cancelled_llm_client_prompt_and_stream_short_circuit() {
        use futures::StreamExt;

        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "ollama"),
            ("LLM_MODEL", "qwen2.5:7b"),
        ]))
        .expect("ollama config");
        let client = LlmClient::from_config(&cfg).expect("client should construct");

        let cancel = CancellationToken::new();
        cancel.cancel();

        let prompt_err = client
            .prompt(None, "hello", cancel.clone())
            .await
            .expect_err("pre-cancelled prompt should fail");
        assert!(matches!(prompt_err, AgentError::Stream(_)));

        let chunks: Vec<StreamChunk> = client
            .stream_prompt(Some("sys"), "hello", cancel)
            .await
            .expect("pre-cancelled stream_prompt should return stream")
            .map(|res| res.expect("chunk ok"))
            .collect()
            .await;
        assert_eq!(chunks, vec![StreamChunk::Cancelled]);
    }

    #[tokio::test]
    async fn empty_or_whitespace_prompt_is_rejected() {
        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "ollama"),
            ("LLM_MODEL", "qwen2.5:7b"),
        ]))
        .expect("ollama config");
        let client = LlmClient::from_config(&cfg).expect("client should construct");

        for empty in ["", "   \t\n "] {
            let prompt_err = client
                .prompt(None, empty, CancellationToken::new())
                .await
                .expect_err("empty prompt should be rejected");
            assert!(
                matches!(prompt_err, AgentError::Stream(ref msg) if msg == "prompt must not be empty")
            );

            let stream_res = client
                .stream_prompt(None, empty, CancellationToken::new())
                .await;
            assert!(matches!(
                stream_res,
                Err(AgentError::Stream(ref msg)) if msg == "prompt must not be empty"
            ));
        }
    }

    #[tokio::test]
    async fn llm_client_prompt_and_stream_emit_genai_chat_spans_against_mock_http() {
        use axum::http::header::CONTENT_TYPE;
        use axum::response::IntoResponse;
        use axum::routing::post;
        use axum::{Json, Router};
        use futures::StreamExt;
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry::Value;
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
        use serde_json::json;
        use tokio::net::TcpListener;
        use tracing_subscriber::layer::SubscriberExt;

        use crate::telemetry::{
            GEN_AI_OPERATION_NAME, GEN_AI_PROVIDER_NAME, GEN_AI_REQUEST_MODEL,
            GEN_AI_RESPONSE_COMPLETION_TOKENS, GEN_AI_SERVER_TTFT_US, GEN_AI_SYSTEM,
        };

        let app = Router::new().route(
            "/v1/chat/completions",
            post(|Json(body): Json<serde_json::Value>| async move {
                let is_stream = body
                    .get("stream")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_stream {
                    let sse = concat!(
                        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"stream!\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n\n"
                    );
                    ([(CONTENT_TYPE, "text/event-stream")], sse).into_response()
                } else {
                    Json(json!({
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello from mock OpenAI"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 4,
                            "total_tokens": 9
                        }
                    }))
                    .into_response()
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "openai"),
            ("LLM_MODEL", "gpt-4o-mini"),
            ("OPENAI_API_KEY", "sk-test"),
            ("LLM_API_BASE", &format!("http://{addr}/v1")),
        ]))
        .expect("openai config with mock base");
        let client = LlmClient::from_config(&cfg).expect("LlmClient should construct");

        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("ferrous-agent-provider-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        let _default_guard = tracing::subscriber::set_default(subscriber);

        let reply = client
            .prompt(Some("Be concise."), "hi", CancellationToken::new())
            .await
            .expect("mock prompt should succeed");
        assert_eq!(reply, "Hello from mock OpenAI");

        let chunks: Vec<StreamChunk> = client
            .stream_prompt(Some("Be concise."), "hi stream", CancellationToken::new())
            .await
            .expect("mock stream_prompt should succeed")
            .map(|r| r.expect("chunk ok"))
            .collect()
            .await;
        assert_eq!(
            chunks,
            vec![
                StreamChunk::Token("Hello ".to_string()),
                StreamChunk::Token("stream!".to_string()),
                StreamChunk::Done,
            ]
        );

        let _ = provider.force_flush();
        let spans = exporter.get_finished_spans().unwrap();
        let genai_spans: Vec<_> = spans
            .iter()
            .filter(|s| s.name.as_ref() == "gen_ai.chat")
            .collect();
        assert_eq!(
            genai_spans.len(),
            2,
            "expected chat and stream gen_ai.chat spans"
        );

        let find_attr = |span: &opentelemetry_sdk::trace::SpanData, key: &str| {
            span.attributes
                .iter()
                .find(|kv| kv.key.as_str() == key)
                .map(|kv| kv.value.clone())
        };

        let chat_span = genai_spans
            .iter()
            .find(|s| find_attr(s, GEN_AI_OPERATION_NAME) == Some(Value::String("chat".into())))
            .expect("chat span");
        assert_eq!(
            find_attr(chat_span, GEN_AI_SYSTEM),
            Some(Value::String("rig-core".into()))
        );
        assert_eq!(
            find_attr(chat_span, GEN_AI_REQUEST_MODEL),
            Some(Value::String("gpt-4o-mini".into()))
        );
        assert_eq!(
            find_attr(chat_span, GEN_AI_PROVIDER_NAME),
            Some(Value::String("openai".into()))
        );
        assert_eq!(
            find_attr(chat_span, GEN_AI_RESPONSE_COMPLETION_TOKENS),
            Some(Value::I64(4))
        );
        assert!(
            find_attr(chat_span, GEN_AI_SERVER_TTFT_US).is_none(),
            "non-streaming chat must not record TTFT"
        );

        let stream_span = genai_spans
            .iter()
            .find(|s| find_attr(s, GEN_AI_OPERATION_NAME) == Some(Value::String("stream".into())))
            .expect("stream span");
        assert_eq!(
            find_attr(stream_span, GEN_AI_SYSTEM),
            Some(Value::String("rig-core".into()))
        );
        assert_eq!(
            find_attr(stream_span, GEN_AI_REQUEST_MODEL),
            Some(Value::String("gpt-4o-mini".into()))
        );
        assert_eq!(
            find_attr(stream_span, GEN_AI_RESPONSE_COMPLETION_TOKENS),
            Some(Value::I64(2))
        );
        assert!(
            find_attr(stream_span, GEN_AI_SERVER_TTFT_US).is_some(),
            "streaming call must record TTFT"
        );

        let _ = provider.shutdown();
        server_handle.abort();
    }
}
