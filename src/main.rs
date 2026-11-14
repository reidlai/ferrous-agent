use std::process::ExitCode;

use ferrous_agent::agent::copilot::CopilotAgent;
use ferrous_agent::agent::provider::LlmClient;
use ferrous_agent::config::AgentConfig;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let mut check_config_only = false;
    for arg in std::env::args().skip(1) {
        if arg == "--check-config" {
            check_config_only = true;
        } else {
            eprintln!("Unknown argument: {arg}");
            return ExitCode::FAILURE;
        }
    }

    let config = match AgentConfig::from_env() {
        Ok(cfg) => cfg,
        Err(err) => {
            tracing::error!(error = %err, "failed to load ferrous-agent configuration");
            eprintln!("Configuration error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let client = match LlmClient::from_config(&config) {
        Ok(client) => client,
        Err(err) => {
            tracing::error!(error = %err, "failed to initialize LLM client");
            eprintln!("Provider initialization error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let agent = CopilotAgent::builder(client)
        .preamble("You are FerrousCX Copilot, an enterprise customer experience assistant.")
        .build();

    tracing::info!(
        provider = %agent.provider(),
        model = %agent.model_name(),
        api_base = ?config.api_base,
        check_config_only,
        "ferrous-agent initialized"
    );

    if check_config_only {
        println!(
            "OK: provider={} model={} api_base={}",
            agent.provider(),
            agent.model_name(),
            config.api_base.as_deref().unwrap_or("<provider-default>")
        );
    }

    ExitCode::SUCCESS
}
