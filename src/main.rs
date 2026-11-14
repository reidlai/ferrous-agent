use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};
use ferrous_agent::agent::copilot::CopilotAgent;
use ferrous_agent::agent::provider::{CompletionDispatch, LlmClient};
use ferrous_agent::benchmark::{
    append_delta_batch, results_to_record_batch, serve_sse, BenchmarkRunner, BenchmarkSseHub,
    BenchmarkSuite, DryRunDispatch, DEFAULT_DELTA_DIR,
};
use ferrous_agent::config::AgentConfig;
use ferrous_agent::server::{probe_health, service_router, HealthState};
use ferrous_agent::telemetry;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "ferrous-agent",
    about = "Pure-Rust CX Copilot Agent & In-Process LLM Benchmark Suite"
)]
struct Cli {
    /// Validate environment configuration and provider initialization, then exit.
    #[arg(long)]
    check_config: bool,

    /// Probe the local `/health` endpoint and exit 0 if healthy, 1 otherwise.
    #[arg(long)]
    healthcheck: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the native in-process LLM benchmark suite.
    Benchmark(BenchmarkArgs),
    /// Run the HTTP service exposing `/health` and benchmark SSE streams.
    Serve(ServeArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    /// Port to listen on when `--bind` is not specified.
    #[arg(long, env = "PORT", default_value_t = 8081, value_parser = parse_port_env)]
    port: u16,

    /// Optional socket address overriding `0.0.0.0:<port>`.
    #[arg(long)]
    bind: Option<SocketAddr>,
}

fn parse_port_env(raw: &str) -> Result<u16, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Ok(8081)
    } else {
        trimmed
            .parse::<u16>()
            .map_err(|err| format!("Invalid PORT value '{raw}': {err}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Delta,
    Json,
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    /// Benchmark suite to execute.
    #[arg(long, value_enum, default_value_t = BenchmarkSuite::All)]
    suite: BenchmarkSuite,

    /// Output destination format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Delta)]
    output: OutputFormat,

    /// Target Delta Lake directory for Arrow RecordBatch / Parquet persistence.
    #[arg(long, default_value = DEFAULT_DELTA_DIR)]
    delta_dir: PathBuf,

    /// Execute offline using a deterministic synthetic DryRunDispatch.
    #[arg(long)]
    dry_run: bool,

    /// Optional socket address to serve live benchmark SSE telemetry (`GET /api/v1/benchmarks/{run_id}/stream`).
    #[arg(long)]
    sse_bind: Option<SocketAddr>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Skip OTLP exporter initialization during short-lived `--healthcheck` probes.
    let _telemetry_guard = (!cli.healthcheck).then(telemetry::init_telemetry);

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("Failed to initialize Tokio runtime: {err}");
            return ExitCode::FAILURE;
        }
    };

    runtime.block_on(async move {
        if cli.healthcheck {
            return run_healthcheck_cli().await;
        }
        match cli.command {
            Some(Commands::Benchmark(args)) => run_benchmark_cli(args).await,
            Some(Commands::Serve(args)) => run_serve_cli(args).await,
            None => run_agent_init(cli.check_config),
        }
    })
}

async fn run_healthcheck_cli() -> ExitCode {
    let port: u16 = match std::env::var("PORT") {
        Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("Invalid PORT value '{raw}': {err}");
                return ExitCode::FAILURE;
            }
        },
        _ => 8081,
    };
    let url = format!("http://127.0.0.1:{port}/health");
    match probe_health(&url).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

async fn run_serve_cli(args: ServeArgs) -> ExitCode {
    let addr = args
        .bind
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], args.port)));

    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(err) => {
            tracing::error!(bind_addr = %addr, error = %err, "failed to bind HTTP service listener");
            eprintln!("Failed to bind HTTP service on {addr}: {err}");
            return ExitCode::FAILURE;
        }
    };
    let bound_addr = listener.local_addr().unwrap_or(addr);

    let health_state = HealthState::from_env();
    let sse_hub = BenchmarkSseHub::new();
    let router = service_router(health_state.clone(), sse_hub);

    tracing::info!(
        bind_addr = %bound_addr,
        "ferrous-agent HTTP service listening"
    );

    if let Err(err) = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            health_state.set_ready(false);
        })
        .await
    {
        tracing::error!(error = %err, "ferrous-agent HTTP service exited with error");
        eprintln!("HTTP service error: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received, draining connections");
}

fn run_agent_init(check_config_only: bool) -> ExitCode {
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

async fn run_benchmark_cli(args: BenchmarkArgs) -> ExitCode {
    let dispatch: Arc<dyn CompletionDispatch> = if args.dry_run {
        Arc::new(DryRunDispatch::default())
    } else {
        let config = match AgentConfig::from_env() {
            Ok(cfg) => cfg,
            Err(err) => {
                tracing::error!(error = %err, "failed to load ferrous-agent configuration");
                eprintln!("Configuration error: {err}");
                return ExitCode::FAILURE;
            }
        };
        match LlmClient::from_config(&config) {
            Ok(client) => Arc::new(client),
            Err(err) => {
                tracing::error!(error = %err, "failed to initialize LLM client");
                eprintln!("Provider initialization error: {err}");
                return ExitCode::FAILURE;
            }
        }
    };

    let run_id = Uuid::now_v7().to_string();
    let sse_server = if let Some(addr) = args.sse_bind {
        let listener = match TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(err) => {
                eprintln!("Failed to bind SSE listener on {addr}: {err}");
                return ExitCode::FAILURE;
            }
        };
        let bound_addr = listener.local_addr().unwrap_or(addr);
        let hub = BenchmarkSseHub::new();
        let tx = hub.register_run(&run_id).await;
        tracing::info!(
            sse_url = %format!("http://{bound_addr}/api/v1/benchmarks/{run_id}/stream"),
            "serving live benchmark SSE telemetry"
        );
        let handle = tokio::spawn(async move {
            let _ = serve_sse(listener, hub).await;
        });
        Some((tx, handle))
    } else {
        None
    };

    let runner = match sse_server {
        Some((ref tx, _)) => BenchmarkRunner::with_broadcaster(dispatch, tx.clone()),
        None => BenchmarkRunner::new(dispatch),
    };
    let runner = match runner {
        Ok(runner) => runner.with_dry_run(args.dry_run),
        Err(err) => {
            eprintln!("Benchmark initialization error: {err}");
            return ExitCode::FAILURE;
        }
    };

    let report = runner
        .run_suite_with_id(&run_id, args.suite, CancellationToken::new())
        .await;

    if let Some((_, handle)) = sse_server {
        handle.abort();
    }

    match args.output {
        OutputFormat::Delta => {
            let batch = match results_to_record_batch(&report.results) {
                Ok(b) => b,
                Err(err) => {
                    eprintln!("Failed to build Arrow RecordBatch: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let version = match append_delta_batch(&args.delta_dir, &batch) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!(
                        "Failed to persist Delta batch to {}: {err}",
                        args.delta_dir.display()
                    );
                    return ExitCode::FAILURE;
                }
            };
            println!(
                "Benchmark run={} suite={} passed={}/{} delta_dir={} delta_version={}",
                report.run_id,
                report.suite,
                report.passed_cases,
                report.total_cases,
                args.delta_dir.display(),
                version
            );
        }
        OutputFormat::Json => match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(err) => {
                eprintln!("Failed to serialize benchmark report: {err}");
                return ExitCode::FAILURE;
            }
        },
    }

    if report.failed_cases > 0 || report.cancelled {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
