use std::collections::HashMap;
use std::fmt;

use crate::agent::AgentError;

/// Comma-separated list of supported `LLM_PROVIDER` values.
pub const SUPPORTED_PROVIDERS: &str =
    "ollama, vllm, huggingface, openrouter, groq, anthropic, openai, gemini";

/// Default base URL for a local Ollama daemon.
pub const DEFAULT_OLLAMA_API_BASE: &str = "http://localhost:11434";

/// Default base URL for a local vLLM OpenAI-compatible server.
pub const DEFAULT_VLLM_API_BASE: &str = "http://localhost:8000/v1";

/// Wrapper for API keys and tokens that redacts its contents in `Debug` and `Display`.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    /// Wrap a raw secret string.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Expose the inner secret for authenticated HTTP requests.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// Supported LLM provider backends across local, serverless, and frontier tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Ollama,
    Vllm,
    HuggingFace,
    OpenRouter,
    Groq,
    Anthropic,
    OpenAi,
    Gemini,
}

impl ProviderKind {
    /// Parse a provider identifier from `LLM_PROVIDER` (case-insensitive).
    pub fn parse(raw: &str) -> Result<Self, AgentError> {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "ollama" => Ok(Self::Ollama),
            "vllm" => Ok(Self::Vllm),
            "huggingface" | "hf" => Ok(Self::HuggingFace),
            "openrouter" => Ok(Self::OpenRouter),
            "groq" => Ok(Self::Groq),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAi),
            "gemini" | "google" => Ok(Self::Gemini),
            _ => Err(AgentError::UnsupportedProvider {
                provider: raw.trim().to_string(),
                supported: SUPPORTED_PROVIDERS,
            }),
        }
    }

    /// Canonical lowercase name of this provider.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Vllm => "vllm",
            Self::HuggingFace => "huggingface",
            Self::OpenRouter => "openrouter",
            Self::Groq => "groq",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gemini => "gemini",
        }
    }

    /// Provider-specific credential environment variable, if any.
    pub fn credential_env_var(&self) -> Option<&'static str> {
        match self {
            Self::Ollama => None,
            Self::Vllm | Self::OpenAi => Some("OPENAI_API_KEY"),
            Self::HuggingFace => Some("HF_TOKEN"),
            Self::OpenRouter => Some("OPENROUTER_API_KEY"),
            Self::Groq => Some("GROQ_API_KEY"),
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::Gemini => Some("GEMINI_API_KEY"),
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Runtime LLM configuration resolved from environment variables.
#[derive(Clone, PartialEq, Eq)]
pub struct AgentConfig {
    pub provider: ProviderKind,
    pub model: String,
    pub api_base: Option<String>,
    api_key: Option<SecretString>,
}

impl fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted_key = if self.api_key.is_some() {
            "[REDACTED]"
        } else {
            "<none>"
        };
        f.debug_struct("AgentConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_base", &self.api_base)
            .field("api_key", &redacted_key)
            .finish()
    }
}

impl AgentConfig {
    /// Load configuration from the current process environment, ignoring non-UTF-8 entries.
    pub fn from_env() -> Result<Self, AgentError> {
        let env: HashMap<String, String> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Self::from_map(&env)
    }

    /// Load configuration from an explicit key-value map (for hermetic testing).
    pub fn from_map(env: &HashMap<String, String>) -> Result<Self, AgentError> {
        let raw_provider = non_empty(env, "LLM_PROVIDER").unwrap_or_default();
        let provider = ProviderKind::parse(&raw_provider)?;

        let model = non_empty(env, "LLM_MODEL").ok_or(AgentError::MissingConfig("LLM_MODEL"))?;

        let explicit_base = non_empty(env, "LLM_API_BASE");
        let api_base = match provider {
            ProviderKind::Ollama => {
                Some(explicit_base.unwrap_or_else(|| DEFAULT_OLLAMA_API_BASE.to_string()))
            }
            ProviderKind::Vllm => {
                Some(explicit_base.unwrap_or_else(|| DEFAULT_VLLM_API_BASE.to_string()))
            }
            _ => explicit_base,
        };

        let explicit_key = non_empty(env, "LLM_API_KEY").or_else(|| {
            provider
                .credential_env_var()
                .and_then(|k| non_empty(env, k))
        });

        let api_key = match provider {
            ProviderKind::Ollama => explicit_key.map(SecretString::new),
            ProviderKind::Vllm => Some(SecretString::new(
                explicit_key.unwrap_or_else(|| "dummy".to_string()),
            )),
            _ => {
                let key = explicit_key.ok_or_else(|| AgentError::MissingCredential {
                    provider: provider.as_str().to_string(),
                    env_var: provider.credential_env_var().unwrap_or("LLM_API_KEY"),
                })?;
                Some(SecretString::new(key))
            }
        };

        Ok(Self {
            provider,
            model,
            api_base,
            api_key,
        })
    }

    /// Access the redacted credential wrapper, if configured.
    pub fn api_key(&self) -> Option<&SecretString> {
        self.api_key.as_ref()
    }

    /// Expose the raw credential string for provider client construction.
    pub fn api_key_str(&self) -> Option<&str> {
        self.api_key.as_ref().map(SecretString::expose)
    }
}

fn non_empty(env: &HashMap<String, String>, key: &str) -> Option<String> {
    env.get(key)
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn ollama_defaults_base_url_and_requires_no_key() {
        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "ollama"),
            ("LLM_MODEL", "qwen2.5:7b"),
        ]))
        .expect("ollama config should succeed");

        assert_eq!(cfg.provider, ProviderKind::Ollama);
        assert_eq!(cfg.model, "qwen2.5:7b");
        assert_eq!(cfg.api_base.as_deref(), Some(DEFAULT_OLLAMA_API_BASE));
        assert!(cfg.api_key().is_none());
    }

    #[test]
    fn ollama_respects_custom_base_url() {
        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "ollama"),
            ("LLM_MODEL", "qwen2.5:7b"),
            ("LLM_API_BASE", "http://10.0.0.5:11434"),
        ]))
        .expect("ollama config with custom base should succeed");

        assert_eq!(cfg.api_base.as_deref(), Some("http://10.0.0.5:11434"));
    }

    #[test]
    fn vllm_uses_dummy_key_fallback_only_for_vllm() {
        let vllm_cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "vllm"),
            ("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            ("LLM_API_BASE", "http://localhost:8000/v1"),
        ]))
        .expect("vllm config should succeed without key");

        assert_eq!(vllm_cfg.provider, ProviderKind::Vllm);
        assert_eq!(
            vllm_cfg.api_base.as_deref(),
            Some("http://localhost:8000/v1")
        );
        assert_eq!(vllm_cfg.api_key_str(), Some("dummy"));

        let vllm_default_base = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "vllm"),
            ("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        ]))
        .expect("vllm config should default LLM_API_BASE");
        assert_eq!(
            vllm_default_base.api_base.as_deref(),
            Some(DEFAULT_VLLM_API_BASE)
        );
        assert_eq!(vllm_default_base.api_key_str(), Some("dummy"));

        // openai with LLM_API_BASE must still require a credential
        let openai_err = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "openai"),
            ("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            ("LLM_API_BASE", "http://localhost:8000/v1"),
        ]))
        .expect_err("openai without key must fail");

        assert!(matches!(
            openai_err,
            AgentError::MissingCredential {
                env_var: "OPENAI_API_KEY",
                ..
            }
        ));
    }

    #[test]
    fn serverless_and_frontier_providers_accept_provider_keys_and_llm_api_key() {
        let cases = [
            ("huggingface", "Qwen/Qwen2.5-7B-Instruct", "HF_TOKEN"),
            ("openrouter", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
            ("groq", "llama-3.1-8b-instant", "GROQ_API_KEY"),
            (
                "anthropic",
                "claude-3-5-sonnet-20241022",
                "ANTHROPIC_API_KEY",
            ),
            ("openai", "gpt-4o", "OPENAI_API_KEY"),
            ("gemini", "gemini-1.5-pro", "GEMINI_API_KEY"),
        ];

        for (provider, model, env_var) in cases {
            let secret = format!("secret-for-{provider}");
            let cfg_specific = AgentConfig::from_map(&map(&[
                ("LLM_PROVIDER", provider),
                ("LLM_MODEL", model),
                (env_var, &secret),
            ]))
            .expect("provider-specific key should succeed");
            assert_eq!(cfg_specific.api_key_str(), Some(secret.as_str()));

            let cfg_generic = AgentConfig::from_map(&map(&[
                ("LLM_PROVIDER", provider),
                ("LLM_MODEL", model),
                ("LLM_API_KEY", &secret),
            ]))
            .expect("LLM_API_KEY should succeed");
            assert_eq!(cfg_generic.api_key_str(), Some(secret.as_str()));
        }
    }

    #[test]
    fn missing_credential_names_env_var_without_secrets() {
        let cases = [
            ("huggingface", "HF_TOKEN"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
        ];

        for (provider, expected_var) in cases {
            let err = AgentConfig::from_map(&map(&[
                ("LLM_PROVIDER", provider),
                ("LLM_MODEL", "some-model"),
            ]))
            .expect_err("missing credential should fail");

            match &err {
                AgentError::MissingCredential { env_var, .. } => {
                    assert_eq!(*env_var, expected_var);
                }
                other => panic!("expected MissingCredential, got {other:?}"),
            }
            assert!(err.to_string().contains(expected_var));
        }
    }

    #[test]
    fn unknown_or_empty_provider_is_rejected() {
        for raw in ["unknown", "", "   "] {
            let err = ProviderKind::parse(raw).expect_err("unknown provider should fail");
            assert!(matches!(err, AgentError::UnsupportedProvider { .. }));
            assert!(err.to_string().contains("ollama"));
        }

        let err = AgentConfig::from_map(&map(&[("LLM_MODEL", "gpt-4o")]))
            .expect_err("unset LLM_PROVIDER should fail");
        assert!(matches!(err, AgentError::UnsupportedProvider { .. }));
    }

    #[test]
    fn missing_or_empty_model_is_rejected() {
        let err = AgentConfig::from_map(&map(&[("LLM_PROVIDER", "ollama")]))
            .expect_err("missing LLM_MODEL should fail");
        assert!(matches!(err, AgentError::MissingConfig("LLM_MODEL")));

        let err_empty =
            AgentConfig::from_map(&map(&[("LLM_PROVIDER", "ollama"), ("LLM_MODEL", "   ")]))
                .expect_err("empty LLM_MODEL should fail");
        assert!(matches!(err_empty, AgentError::MissingConfig("LLM_MODEL")));
    }

    #[test]
    fn debug_and_display_redact_secrets() {
        let secret = "sk-super-secret-token-12345";
        let cfg = AgentConfig::from_map(&map(&[
            ("LLM_PROVIDER", "openai"),
            ("LLM_MODEL", "gpt-4o"),
            ("OPENAI_API_KEY", secret),
        ]))
        .expect("config should succeed");

        let debug_str = format!("{cfg:?}");
        assert!(!debug_str.contains(secret));
        assert!(debug_str.contains("[REDACTED]"));

        let key = cfg.api_key().expect("key should be present");
        assert!(!format!("{key:?}").contains(secret));
        assert!(!format!("{key}").contains(secret));
        assert_eq!(format!("{key}"), "[REDACTED]");
    }

    #[test]
    fn provider_kind_parses_aliases_normalizes_case_and_displays() {
        let cases = [
            ("  HF ", ProviderKind::HuggingFace, "huggingface"),
            ("Claude", ProviderKind::Anthropic, "anthropic"),
            ("GOOGLE\t", ProviderKind::Gemini, "gemini"),
            ("  OlLaMa  ", ProviderKind::Ollama, "ollama"),
            ("VLLM", ProviderKind::Vllm, "vllm"),
            ("OpenRouter", ProviderKind::OpenRouter, "openrouter"),
            ("GROQ", ProviderKind::Groq, "groq"),
            ("OpenAI", ProviderKind::OpenAi, "openai"),
        ];

        for (raw, expected_kind, expected_display) in cases {
            let parsed = ProviderKind::parse(raw).expect("alias/case should parse");
            assert_eq!(parsed, expected_kind);
            assert_eq!(parsed.to_string(), expected_display);
        }
    }
}
