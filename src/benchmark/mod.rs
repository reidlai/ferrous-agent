//! Native in-process LLM benchmark suite (Story 9.4).
//!
//! Provides microsecond TTFT/ITL/TPS telemetry, sub-millisecond `schemars` + `jsonschema`
//! tool-call validation, `$20.00` courtesy-credit and `ContextSanitizer` guardrail assertions,
//! atomic Apache Arrow / Delta Lake persistence, and Axum SSE live streaming.

pub mod cases;
pub mod metrics;
pub mod runner;
pub mod sse;
pub mod storage;

pub use cases::{
    golden_cases, BenchmarkCase, BenchmarkCaseKind, BenchmarkCategory, BenchmarkSuite,
    ToolSchemaValidator, ToolValidationResult,
};
pub use metrics::{collect_stream_metrics, BenchmarkMetrics, StreamTimingCollector};
pub use runner::{
    BenchmarkCaseResult, BenchmarkEvent, BenchmarkRunReport, BenchmarkRunner, DryRunDispatch,
};
pub use sse::{benchmark_sse_router, serve_sse, BenchmarkSseHub};
pub use storage::{
    append_delta_batch, llm_benchmarks_schema, results_to_record_batch, BenchmarkStorageError,
    DEFAULT_DELTA_DIR,
};
