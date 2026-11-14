use async_trait::async_trait;
use rig_core::completion::CompletionRequest;
use rig_core::driver::DynModel;
use rig_core::operation::Completion;
use rig_core::providers::{anthropic, gemini, ollama, openai};
use tokio_util::sync::CancellationToken;

use crate::agent::streaming::{adapt_rig_stream, BoxedChunkStream, StreamChunk};
use crate::agent::AgentError;
use crate::config::{AgentConfig, ProviderKind};

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
        if prompt.trim().is_empty() {
            return Err(AgentError::Stream("prompt must not be empty".to_string()));
        }
        if cancel.is_cancelled() {
            return Err(AgentError::Stream("prompt cancelled".to_string()));
        }

        let request = build_request(system_preamble, prompt);
        let fut = self.model.call(request);

        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                Err(AgentError::Stream("prompt cancelled".to_string()))
            }
            res = fut => {
                let response = res?;
                Ok(response.text())
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

        let request = build_request(system_preamble, prompt);
        let stream = self.model.stream(request)?;
        Ok(adapt_rig_stream(stream, cancel))
    }
}

fn build_request(system_preamble: Option<&str>, prompt: &str) -> CompletionRequest {
    let mut req = CompletionRequest::new(prompt);
    if let Some(preamble) = system_preamble.filter(|p| !p.trim().is_empty()) {
        req = req.preamble(preamble);
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
}
