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
