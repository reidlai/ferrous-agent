use std::process::Command;

#[test]
fn check_config_succeeds_with_clean_stdout_for_ollama() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--check-config")
        .env("LLM_PROVIDER", "ollama")
        .env("LLM_MODEL", "qwen2.5:7b")
        .env_remove("LLM_API_BASE")
        .env_remove("LLM_API_KEY")
        .output()
        .expect("failed to execute ferrous-agent binary");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout
            .trim()
            .starts_with("OK: provider=ollama model=qwen2.5:7b"),
        "unexpected stdout: {stdout:?}"
    );
    assert!(
        !stdout.contains("INFO ferrous_agent"),
        "tracing logs leaked to stdout: {stdout:?}"
    );
    assert!(
        stderr.contains("ferrous-agent initialized"),
        "expected tracing output on stderr, got: {stderr:?}"
    );
}

#[test]
fn check_config_fails_when_required_env_is_missing() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--check-config")
        .env_remove("LLM_PROVIDER")
        .env_remove("LLM_MODEL")
        .env_remove("LLM_API_BASE")
        .env_remove("LLM_API_KEY")
        .output()
        .expect("failed to execute ferrous-agent binary");

    assert!(
        !output.status.success(),
        "expected non-zero exit when LLM_PROVIDER is missing"
    );
}

#[test]
fn unknown_cli_flag_fails_with_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--invalid")
        .env("LLM_PROVIDER", "ollama")
        .env("LLM_MODEL", "qwen2.5:7b")
        .output()
        .expect("failed to execute ferrous-agent binary");

    assert!(
        !output.status.success(),
        "expected non-zero exit for unknown CLI flag"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--invalid"),
        "expected unknown flag in stderr, got: {stderr:?}"
    );
}

#[test]
fn benchmark_suite_all_dry_run_writes_delta_batch_and_prints_summary() {
    let delta_dir = std::env::temp_dir().join(format!("ferrous-cli-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&delta_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("benchmark")
        .arg("--suite")
        .arg("all")
        .arg("--dry-run")
        .arg("--delta-dir")
        .arg(&delta_dir)
        .env_remove("LLM_PROVIDER")
        .env_remove("LLM_MODEL")
        .output()
        .expect("failed to execute ferrous-agent benchmark");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("suite=all") && stdout.contains("delta_version=0"),
        "unexpected benchmark summary stdout: {stdout:?}"
    );

    let log0 = delta_dir
        .join("_delta_log")
        .join("00000000000000000000.json");
    assert!(log0.exists(), "expected Delta log file at {log0:?}");

    let has_parquet = std::fs::read_dir(&delta_dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("parquet"));
    assert!(has_parquet, "expected Parquet data file in {delta_dir:?}");

    let _ = std::fs::remove_dir_all(&delta_dir);
}

#[test]
fn benchmark_invalid_suite_exits_non_zero() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("benchmark")
        .arg("--suite")
        .arg("nonexistent")
        .arg("--dry-run")
        .output()
        .expect("failed to execute ferrous-agent benchmark");

    assert!(
        !output.status.success(),
        "expected non-zero exit for invalid --suite"
    );
}

#[test]
fn benchmark_suite_tools_dry_run_json_outputs_valid_report() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("benchmark")
        .arg("--suite")
        .arg("tools")
        .arg("--dry-run")
        .arg("--output")
        .arg("json")
        .output()
        .expect("failed to execute ferrous-agent benchmark");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout must be valid JSON");

    assert!(parsed.get("run_id").and_then(|v| v.as_str()).is_some());
    assert_eq!(parsed.get("suite").and_then(|v| v.as_str()), Some("tools"));
    assert_eq!(parsed.get("dry_run").and_then(|v| v.as_bool()), Some(true));
    assert!(parsed
        .get("total_cases")
        .and_then(|v| v.as_u64())
        .is_some_and(|n| n > 0));
    assert!(parsed
        .get("results")
        .and_then(|v| v.as_array())
        .is_some_and(|arr| !arr.is_empty()));
}

#[test]
fn healthcheck_succeeds_against_live_local_health_endpoint() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let (port, shutdown_tx, server_handle) = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local_addr").port();
        let app = ferrous_agent::server::service_router(
            ferrous_agent::server::HealthState::new(true, None),
            ferrous_agent::benchmark::BenchmarkSseHub::new(),
        );
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await;
        });
        (port, tx, handle)
    });

    let runtime_thread = std::thread::spawn(move || {
        rt.block_on(async move {
            let _ = server_handle.await;
        });
    });

    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--healthcheck")
        .env("PORT", port.to_string())
        .output()
        .expect("failed to execute ferrous-agent --healthcheck");

    let _ = shutdown_tx.send(());
    let _ = runtime_thread.join();

    assert!(
        output.status.success(),
        "expected exit 0 for --healthcheck against live /health, got {:?}, stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn healthcheck_fails_when_port_is_unreachable() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let closed_port = listener.local_addr().expect("local_addr").port();
    drop(listener);

    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--healthcheck")
        .env("PORT", closed_port.to_string())
        .output()
        .expect("failed to execute ferrous-agent --healthcheck");

    assert!(
        !output.status.success(),
        "expected non-zero exit for --healthcheck on unreachable port {closed_port}"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("healthcheck"),
        "expected healthcheck failure message on stderr, got: {stderr:?}"
    );
}

struct ChildGuard(std::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn serve_process_responds_to_healthcheck_and_exits_cleanly() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);

    let child = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .args(["serve", "--bind", &format!("127.0.0.1:{port}")])
        .env_remove("QDRANT_URL")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn ferrous-agent serve");
    let mut guard = ChildGuard(child);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .expect("build reqwest client");
    let health_url = format!("http://127.0.0.1:{port}/health");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut ready = false;
    while std::time::Instant::now() < deadline {
        if let Some(status) = guard.0.try_wait().expect("poll child status") {
            panic!("ferrous-agent serve exited prematurely with {status}");
        }
        let ok = rt.block_on(async {
            matches!(
                client.get(&health_url).send().await,
                Ok(resp) if resp.status() == reqwest::StatusCode::OK
            )
        });
        if ok {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        ready,
        "timed out waiting for GET {health_url} to return 200 OK"
    );

    let hc_output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--healthcheck")
        .env("PORT", port.to_string())
        .output()
        .expect("execute ferrous-agent --healthcheck");

    assert!(
        hc_output.status.success(),
        "expected exit 0 for --healthcheck against spawned serve, got {:?}, stderr: {}",
        hc_output.status.code(),
        String::from_utf8_lossy(&hc_output.stderr)
    );

    guard.0.kill().expect("terminate ferrous-agent serve");
    let _ = guard.0.wait().expect("wait for ferrous-agent serve");
}

#[test]
fn healthcheck_invalid_port_env_exits_non_zero() {
    let output = Command::new(env!("CARGO_BIN_EXE_ferrous-agent"))
        .arg("--healthcheck")
        .env("PORT", "notaport")
        .output()
        .expect("execute ferrous-agent --healthcheck");

    assert!(
        !output.status.success(),
        "expected non-zero exit for PORT=notaport --healthcheck"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Invalid PORT value"),
        "expected 'Invalid PORT value' on stderr, got: {stderr:?}"
    );
}
