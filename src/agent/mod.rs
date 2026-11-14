pub mod copilot;
pub mod provider;
pub mod streaming;

use thiserror::Error;

/// Errors returned by configuration loading, provider construction, and LLM dispatch.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The configured `LLM_PROVIDER` is empty or not one of the supported providers.
    #[error("unsupported LLM provider '{provider}'; supported providers: {supported}")]
    UnsupportedProvider {
        provider: String,
        supported: &'static str,
    },

    /// A required configuration variable is missing or empty.
    #[error("missing required configuration: {0}")]
    MissingConfig(&'static str),

    /// A provider credential is missing; names the expected env var without exposing secrets.
    #[error("missing credential for provider '{provider}': set LLM_API_KEY or {env_var}")]
    MissingCredential {
        provider: String,
        env_var: &'static str,
    },

    /// Underlying `rig-core` provider error.
    #[error("provider request failed: {0}")]
    Provider(#[source] Box<rig_core::error::ProviderError>),

    /// Stream adaptation or cancellation error.
    #[error("stream error: {0}")]
    Stream(String),
}

impl From<rig_core::error::ProviderError> for AgentError {
    fn from(err: rig_core::error::ProviderError) -> Self {
        Self::Provider(Box::new(err))
    }
}
