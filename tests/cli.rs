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
