//! Asynchronous benchmark runner, `DryRunDispatch`, and broadcast event telemetry.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::agent::provider::CompletionDispatch;
use crate::agent::streaming::{adapt_token_stream, BoxedChunkStream};
use crate::agent::AgentError;
use crate::benchmark::cases::{
    evaluate_guardrail_case, evaluate_tool_case, BenchmarkCase, BenchmarkCaseKind, BenchmarkSuite,
    ToolSchemaValidator,
};
use crate::benchmark::metrics::{collect_stream_metrics, BenchmarkMetrics};
use crate::config::ProviderKind;

/// Default broadcast channel capacity for live benchmark telemetry.
pub const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 256;

/// Result of executing a single [`BenchmarkCase`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkCaseResult {
    pub run_id: String,
    pub case_id: String,
    pub category: String,
    pub case_name: String,
    pub provider: String,
    pub model: String,
    pub dry_run: bool,
    pub passed: bool,
    pub metrics: Option<BenchmarkMetrics>,
    pub total_duration_us: u64,
    pub tool_call_valid: Option<bool>,
    pub guardrail_passed: Option<bool>,
    pub validation_us: Option<u64>,
    pub error: Option<String>,
    pub timestamp_utc: String,
}

/// Aggregated report for a completed benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRunReport {
    pub run_id: String,
    pub suite: BenchmarkSuite,
    pub provider: String,
    pub model: String,
    pub dry_run: bool,
    pub cancelled: bool,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub results: Vec<BenchmarkCaseResult>,
    pub started_at: String,
    pub completed_at: String,
}

/// Telemetry events published over `tokio::sync::broadcast` and Axum SSE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", content = "data", rename_all = "snake_case")]
pub enum BenchmarkEvent {
    TokenChunk {
        run_id: String,
        case_id: String,
        token: String,
        chunk_index: usize,
    },
    CaseCompleted {
        run_id: String,
        result: BenchmarkCaseResult,
    },
    BenchmarkCompleted {
        run_id: String,
        report: BenchmarkRunReport,
    },
}

impl BenchmarkEvent {
    /// SSE `event:` name corresponding to this variant.
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::TokenChunk { .. } => "token_chunk",
            Self::CaseCompleted { .. } => "case_completed",
            Self::BenchmarkCompleted { .. } => "benchmark_completed",
        }
    }

    /// Run identifier associated with this event.
    pub fn run_id(&self) -> &str {
        match self {
            Self::TokenChunk { run_id, .. }
            | Self::CaseCompleted { run_id, .. }
            | Self::BenchmarkCompleted { run_id, .. } => run_id,
        }
    }
}

/// Deterministic offline [`CompletionDispatch`] for CI and `--dry-run` benchmark runs.
#[derive(Debug, Clone)]
pub struct DryRunDispatch {
    provider: ProviderKind,
    model: String,
    chunk_delay: Duration,
}

impl DryRunDispatch {
    /// Create a [`DryRunDispatch`] with a 1 ms inter-token delay.
    pub fn new(provider: ProviderKind, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            chunk_delay: Duration::from_millis(1),
        }
    }

    /// Override the synthetic inter-token delay.
    pub fn with_chunk_delay(mut self, delay: Duration) -> Self {
        self.chunk_delay = delay;
        self
    }
}

impl Default for DryRunDispatch {
    fn default() -> Self {
        Self::new(ProviderKind::Ollama, "dry-run-simulated")
    }
}

#[async_trait]
impl CompletionDispatch for DryRunDispatch {
    fn provider(&self) -> ProviderKind {
        self.provider
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn prompt(
        &self,
        _system_preamble: Option<&str>,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        if cancel.is_cancelled() {
            return Err(AgentError::Stream("prompt cancelled".to_string()));
        }
        Ok(format!(
            "Dry-run analysis for '{}': order verified, policy compliant.",
            prompt.chars().take(48).collect::<String>()
        ))
    }

    async fn stream_prompt(
        &self,
        _system_preamble: Option<&str>,
        _prompt: &str,
        cancel: CancellationToken,
    ) -> Result<BoxedChunkStream, AgentError> {
        let tokens = vec![
            "Checking ".to_string(),
            "order ".to_string(),
            "status, ".to_string(),
            "tracking ".to_string(),
            "SLA, ".to_string(),
            "and ".to_string(),
            "HITL ".to_string(),
            "policy.".to_string(),
        ];
        let delay = self.chunk_delay;
        let stream = futures::stream::unfold(
            (tokens.into_iter(), delay),
            |(mut iter, delay)| async move {
                let next_tok = iter.next()?;
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                Some((Ok(Some(next_tok)), (iter, delay)))
            },
        );
        Ok(adapt_token_stream(stream, cancel))
    }
}

/// Asynchronous in-process benchmark runner.
pub struct BenchmarkRunner {
    dispatch: Arc<dyn CompletionDispatch>,
    validator: ToolSchemaValidator,
    tx: broadcast::Sender<BenchmarkEvent>,
    dry_run: bool,
}

impl BenchmarkRunner {
    /// Construct a [`BenchmarkRunner`] with a fresh broadcast channel.
    pub fn new(dispatch: Arc<dyn CompletionDispatch>) -> Result<Self, String> {
        let (tx, _rx) = broadcast::channel(DEFAULT_EVENT_CHANNEL_CAPACITY);
        Self::with_broadcaster(dispatch, tx)
    }

    /// Construct a [`BenchmarkRunner`] publishing events to an existing broadcast sender.
    pub fn with_broadcaster(
        dispatch: Arc<dyn CompletionDispatch>,
        tx: broadcast::Sender<BenchmarkEvent>,
    ) -> Result<Self, String> {
        let validator = ToolSchemaValidator::new()?;
        Ok(Self {
            dispatch,
            validator,
            tx,
            dry_run: false,
        })
    }

    /// Mark whether this runner is executing in `--dry-run` mode.
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Subscribe to live [`BenchmarkEvent`]s emitted by this runner.
    pub fn subscribe(&self) -> broadcast::Receiver<BenchmarkEvent> {
        self.tx.subscribe()
    }

    /// Access the underlying broadcast sender.
    pub fn sender(&self) -> &broadcast::Sender<BenchmarkEvent> {
        &self.tx
    }

    /// Execute all cases in `suite` under a newly generated UUIDv7 `run_id`.
    pub async fn run_suite(
        &self,
        suite: BenchmarkSuite,
        cancel: CancellationToken,
    ) -> BenchmarkRunReport {
        let run_id = Uuid::now_v7().to_string();
        self.run_suite_with_id(run_id, suite, cancel).await
    }

    /// Execute all cases in `suite` under an explicit `run_id`.
    pub async fn run_suite_with_id(
        &self,
        run_id: impl Into<String>,
        suite: BenchmarkSuite,
        cancel: CancellationToken,
    ) -> BenchmarkRunReport {
        let run_id = run_id.into();
        let started_at = Utc::now().to_rfc3339();
        let cases = suite.cases();
        let mut results = Vec::with_capacity(cases.len());
        let mut cancelled = false;

        for case in cases {
            if cancel.is_cancelled() {
                cancelled = true;
                break;
            }
            let res = self.run_case(&run_id, &case, cancel.clone()).await;
            let _ = self.tx.send(BenchmarkEvent::CaseCompleted {
                run_id: run_id.clone(),
                result: res.clone(),
            });
            results.push(res);
            if cancel.is_cancelled() {
                cancelled = true;
                break;
            }
        }

        let passed_cases = results.iter().filter(|r| r.passed).count();
        let failed_cases = results.len().saturating_sub(passed_cases);
        let completed_at = Utc::now().to_rfc3339();

        let (provider, model) = if self.dry_run {
            ("dry-run".to_string(), "synthetic-dry-run".to_string())
        } else {
            (
                self.dispatch.provider().as_str().to_string(),
                self.dispatch.model_name().to_string(),
            )
        };

        let report = BenchmarkRunReport {
            run_id: run_id.clone(),
            suite,
            provider,
            model,
            dry_run: self.dry_run,
            cancelled,
            total_cases: results.len(),
            passed_cases,
            failed_cases,
            results,
            started_at,
            completed_at,
        };

        let _ = self.tx.send(BenchmarkEvent::BenchmarkCompleted {
            run_id,
            report: report.clone(),
        });

        report
    }

    async fn run_case(
        &self,
        run_id: &str,
        case: &BenchmarkCase,
        cancel: CancellationToken,
    ) -> BenchmarkCaseResult {
        let (provider, model) = if self.dry_run {
            ("dry-run".to_string(), "synthetic-dry-run".to_string())
        } else {
            (
                self.dispatch.provider().as_str().to_string(),
                self.dispatch.model_name().to_string(),
            )
        };
        let timestamp_utc = Utc::now().to_rfc3339();

        match &case.kind {
            BenchmarkCaseKind::LatencyStream {
                system_preamble,
                prompt,
            } => {
                let stream_res = self
                    .dispatch
                    .stream_prompt(system_preamble.as_deref(), prompt, cancel)
                    .await;

                match stream_res {
                    Ok(stream) => {
                        let mut chunk_index = 0usize;
                        let collected = collect_stream_metrics(stream, |token| {
                            let _ = self.tx.send(BenchmarkEvent::TokenChunk {
                                run_id: run_id.to_string(),
                                case_id: case.id.clone(),
                                token: token.to_string(),
                                chunk_index,
                            });
                            chunk_index += 1;
                        })
                        .await;

                        match collected {
                            Ok((text, metrics)) => {
                                let passed = metrics.token_count > 0 && !text.trim().is_empty();
                                let error = (!passed)
                                    .then(|| "latency stream produced empty output".to_string());
                                let total_duration_us = metrics.total_duration_us;
                                BenchmarkCaseResult {
                                    run_id: run_id.to_string(),
                                    case_id: case.id.clone(),
                                    category: case.category.as_str().to_string(),
                                    case_name: case.name.clone(),
                                    provider,
                                    model,
                                    dry_run: self.dry_run,
                                    passed,
                                    metrics: Some(metrics),
                                    total_duration_us,
                                    tool_call_valid: None,
                                    guardrail_passed: None,
                                    validation_us: None,
                                    error,
                                    timestamp_utc,
                                }
                            }
                            Err(err) => BenchmarkCaseResult {
                                run_id: run_id.to_string(),
                                case_id: case.id.clone(),
                                category: case.category.as_str().to_string(),
                                case_name: case.name.clone(),
                                provider,
                                model,
                                dry_run: self.dry_run,
                                passed: false,
                                metrics: None,
                                total_duration_us: 0,
                                tool_call_valid: None,
                                guardrail_passed: None,
                                validation_us: None,
                                error: Some(err.to_string()),
                                timestamp_utc,
                            },
                        }
                    }
                    Err(err) => BenchmarkCaseResult {
                        run_id: run_id.to_string(),
                        case_id: case.id.clone(),
                        category: case.category.as_str().to_string(),
                        case_name: case.name.clone(),
                        provider,
                        model,
                        dry_run: self.dry_run,
                        passed: false,
                        metrics: None,
                        total_duration_us: 0,
                        tool_call_valid: None,
                        guardrail_passed: None,
                        validation_us: None,
                        error: Some(err.to_string()),
                        timestamp_utc,
                    },
                }
            }
            BenchmarkCaseKind::ToolCallValidation {
                tool_name,
                args_json,
                expect_valid,
            } => {
                let (val_res, outcome) =
                    evaluate_tool_case(&self.validator, tool_name, args_json, *expect_valid);
                BenchmarkCaseResult {
                    run_id: run_id.to_string(),
                    case_id: case.id.clone(),
                    category: case.category.as_str().to_string(),
                    case_name: case.name.clone(),
                    provider,
                    model,
                    dry_run: self.dry_run,
                    passed: outcome.is_ok(),
                    metrics: None,
                    total_duration_us: val_res.validation_us,
                    tool_call_valid: Some(val_res.valid),
                    guardrail_passed: None,
                    validation_us: Some(val_res.validation_us),
                    error: outcome.err(),
                    timestamp_utc,
                }
            }
            BenchmarkCaseKind::GuardrailCreditCeiling { .. }
            | BenchmarkCaseKind::GuardrailPiiMasking { .. }
            | BenchmarkCaseKind::GuardrailIngressBlock { .. } => {
                let start = tokio::time::Instant::now();
                let outcome = evaluate_guardrail_case(&case.kind);
                let elapsed_us = start.elapsed().as_micros() as u64;
                let passed = outcome.is_ok();
                BenchmarkCaseResult {
                    run_id: run_id.to_string(),
                    case_id: case.id.clone(),
                    category: case.category.as_str().to_string(),
                    case_name: case.name.clone(),
                    provider,
                    model,
                    dry_run: self.dry_run,
                    passed,
                    metrics: None,
                    total_duration_us: elapsed_us,
                    tool_call_valid: None,
                    guardrail_passed: Some(passed),
                    validation_us: None,
                    error: outcome.err(),
                    timestamp_utc,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dry_run_suite_all_passes_and_publishes_events() {
        let dispatch = Arc::new(DryRunDispatch::default().with_chunk_delay(Duration::ZERO));
        let runner = BenchmarkRunner::new(dispatch).expect("runner should initialize");
        let mut rx = runner.subscribe();

        let report = runner
            .run_suite_with_id("run-test-1", BenchmarkSuite::All, CancellationToken::new())
            .await;

        assert_eq!(report.run_id, "run-test-1");
        assert_eq!(report.failed_cases, 0);
        assert_eq!(report.passed_cases, report.total_cases);
        assert!(report.total_cases >= 11);

        let mut saw_token = false;
        let mut case_completed_count = 0usize;
        let mut saw_completed = false;

        while let Ok(ev) = rx.try_recv() {
            match ev {
                BenchmarkEvent::TokenChunk { .. } => saw_token = true,
                BenchmarkEvent::CaseCompleted { .. } => case_completed_count += 1,
                BenchmarkEvent::BenchmarkCompleted { .. } => saw_completed = true,
            }
        }

        assert!(saw_token, "expected at least one TokenChunk event");
        assert_eq!(case_completed_count, report.total_cases);
        assert!(saw_completed, "expected BenchmarkCompleted event");
    }

    struct EmptyStreamDispatch;

    #[async_trait]
    impl CompletionDispatch for EmptyStreamDispatch {
        fn provider(&self) -> ProviderKind {
            ProviderKind::Ollama
        }

        fn model_name(&self) -> &str {
            "empty-stub"
        }

        async fn prompt(
            &self,
            _system_preamble: Option<&str>,
            _prompt: &str,
            _cancel: CancellationToken,
        ) -> Result<String, AgentError> {
            Ok(String::new())
        }

        async fn stream_prompt(
            &self,
            _system_preamble: Option<&str>,
            _prompt: &str,
            cancel: CancellationToken,
        ) -> Result<BoxedChunkStream, AgentError> {
            let empty = futures::stream::iter(Vec::<Result<Option<String>, AgentError>>::new());
            Ok(adapt_token_stream(empty, cancel))
        }
    }

    #[tokio::test]
    async fn empty_stream_dispatch_marks_latency_cases_failed() {
        let runner = BenchmarkRunner::new(Arc::new(EmptyStreamDispatch)).unwrap();
        let report = runner
            .run_suite_with_id(
                "run-fail",
                BenchmarkSuite::Latency,
                CancellationToken::new(),
            )
            .await;

        assert!(!report.dry_run);
        assert!(!report.cancelled);
        assert!(report.failed_cases > 0);
        assert!(report.results.iter().all(|r| !r.passed));
    }

    #[tokio::test]
    async fn cancelled_token_stops_suite_early_and_sets_cancelled_flag() {
        let dispatch = Arc::new(DryRunDispatch::default().with_chunk_delay(Duration::ZERO));
        let runner = BenchmarkRunner::new(dispatch).unwrap().with_dry_run(true);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let report = runner
            .run_suite_with_id("run-cancel", BenchmarkSuite::All, cancel)
            .await;

        assert!(report.dry_run);
        assert!(report.cancelled);
        assert_eq!(report.total_cases, 0);
        assert_eq!(report.provider, "dry-run");
        assert_eq!(report.model, "synthetic-dry-run");
    }
}
