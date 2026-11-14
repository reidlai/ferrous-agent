//! OpenTelemetry GenAI Semantic Conventions & W3C Trace Context propagation.

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use opentelemetry::global;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer as _};

use crate::agent::streaming::{BoxedChunkStream, StreamChunk};
use crate::agent::AgentError;

/// OpenTelemetry GenAI semantic convention attribute keys.
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";
pub const GEN_AI_SYSTEM_RIG_CORE: &str = "rig-core";
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
pub const GEN_AI_RESPONSE_COMPLETION_TOKENS: &str = "gen_ai.response.completion_tokens";
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
pub const GEN_AI_SERVER_TTFT_US: &str = "gen_ai.server.time_to_first_token_us";

/// Guard that flushes and shuts down the OpenTelemetry `SdkTracerProvider` on drop.
pub struct TelemetryGuard {
    provider: Option<SdkTracerProvider>,
}

impl TelemetryGuard {
    /// Construct a no-op guard when OpenTelemetry export is disabled.
    pub fn disabled() -> Self {
        Self { provider: None }
    }

    /// Returns `true` if an active `SdkTracerProvider` is configured.
    pub fn is_enabled(&self) -> bool {
        self.provider.is_some()
    }

    /// Flush pending spans and shut down the tracer provider.
    pub fn shutdown(&mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.force_flush();
            let _ = provider.shutdown();
        }
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Initialize tracing and OpenTelemetry from process environment variables.
///
/// Registers [`TraceContextPropagator`] globally for W3C `traceparent` / `tracestate`
/// propagation. OTLP span export is enabled only when `OTEL_EXPORTER_OTLP_ENDPOINT` is
/// non-empty and `OTEL_SDK_DISABLED` is not `"true"`.
pub fn init_telemetry() -> TelemetryGuard {
    let env: HashMap<String, String> = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .collect();
    init_telemetry_from_map(&env)
}

/// Initialize tracing and OpenTelemetry from an explicit environment map.
pub fn init_telemetry_from_map(env: &HashMap<String, String>) -> TelemetryGuard {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let provider = build_otlp_provider_from_map(env);
    let env_filter = env
        .get("RUST_LOG")
        .and_then(|val| EnvFilter::try_new(val).ok())
        .or_else(|| EnvFilter::try_from_default_env().ok())
        .unwrap_or_else(|| EnvFilter::new("info"));

    let console_json = env
        .get("OTEL_TRACES_EXPORTER")
        .map(|v| v.trim().eq_ignore_ascii_case("console"))
        .unwrap_or(false);

    let fmt_layer = if console_json {
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .json()
            .boxed()
    } else {
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .boxed()
    };

    if let Some(sdk_provider) = provider {
        let tracer = sdk_provider.tracer("ferrous-agent");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        if tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otel_layer)
            .try_init()
            .is_ok()
        {
            global::set_tracer_provider(sdk_provider.clone());
            TelemetryGuard {
                provider: Some(sdk_provider),
            }
        } else {
            let _ = sdk_provider.shutdown();
            TelemetryGuard::disabled()
        }
    } else {
        let _ = tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init();
        TelemetryGuard::disabled()
    }
}

/// Determine whether OTLP export should be enabled given an environment map.
pub fn is_otlp_enabled(env: &HashMap<String, String>) -> bool {
    let disabled = env
        .get("OTEL_SDK_DISABLED")
        .map(|v| v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if disabled {
        return false;
    }

    if let Some(exporter) = env.get("OTEL_TRACES_EXPORTER") {
        let normalized = exporter.trim();
        if normalized.eq_ignore_ascii_case("none") || normalized.eq_ignore_ascii_case("console") {
            return false;
        }
    }

    env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

fn normalize_otlp_traces_endpoint(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1/traces") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1/traces")
    }
}

fn build_otlp_provider_from_map(env: &HashMap<String, String>) -> Option<SdkTracerProvider> {
    if !is_otlp_enabled(env) {
        return None;
    }

    let raw_endpoint = env.get("OTEL_EXPORTER_OTLP_ENDPOINT")?.trim();
    let endpoint = normalize_otlp_traces_endpoint(raw_endpoint);
    let exporter = match opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .build()
    {
        Ok(exporter) => exporter,
        Err(err) => {
            eprintln!("Warning: failed to build OTLP SpanExporter: {err}");
            return None;
        }
    };

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    Some(provider)
}

/// Create a `tracing::Span` tagged with standard OpenTelemetry GenAI Semantic Conventions.
pub fn genai_chat_span(provider: &str, model: &str, operation: &str) -> tracing::Span {
    tracing::info_span!(
        "gen_ai.chat",
        gen_ai.system = GEN_AI_SYSTEM_RIG_CORE,
        gen_ai.request.model = %model,
        gen_ai.provider.name = %provider,
        gen_ai.operation.name = %operation,
        gen_ai.response.completion_tokens = tracing::field::Empty,
        gen_ai.usage.output_tokens = tracing::field::Empty,
        gen_ai.server.time_to_first_token_us = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    )
}

/// Record time-to-first-token (TTFT) in microseconds on a GenAI span.
pub fn record_ttft(span: &tracing::Span, elapsed: Duration) {
    let micros = i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX);
    span.record(GEN_AI_SERVER_TTFT_US, micros);
}

/// Record completion token count on a GenAI span.
pub fn record_completion_tokens(span: &tracing::Span, tokens: u64) {
    let tokens_i64 = i64::try_from(tokens).unwrap_or(i64::MAX);
    span.record(GEN_AI_RESPONSE_COMPLETION_TOKENS, tokens_i64);
    span.record(GEN_AI_USAGE_OUTPUT_TOKENS, tokens_i64);
}

/// Estimate token count from generated text for providers/streams without explicit usage metadata.
pub fn estimate_token_count(text: &str) -> u64 {
    let words = text.split_whitespace().count() as u64;
    if words == 0 && !text.is_empty() {
        1
    } else {
        words
    }
}

/// Wrap a [`BoxedChunkStream`] so TTFT and completion token counts are recorded on `span`.
///
/// The span is held open until the stream completes (`Done`, `Cancelled`, `Err`) or is dropped.
pub fn instrument_chunk_stream(
    stream: BoxedChunkStream,
    span: tracing::Span,
    started_at: Instant,
) -> BoxedChunkStream {
    Box::pin(InstrumentedChunkStream {
        inner: stream,
        span: Some(span),
        started_at,
        ttft_recorded: false,
        completion_tokens: 0,
    })
}

struct InstrumentedChunkStream {
    inner: BoxedChunkStream,
    span: Option<tracing::Span>,
    started_at: Instant,
    ttft_recorded: bool,
    completion_tokens: u64,
}

impl InstrumentedChunkStream {
    fn finish_span(&mut self) {
        if let Some(span) = self.span.take() {
            record_completion_tokens(&span, self.completion_tokens);
        }
    }
}

impl Drop for InstrumentedChunkStream {
    fn drop(&mut self) {
        self.finish_span();
    }
}

impl Stream for InstrumentedChunkStream {
    type Item = Result<StreamChunk, AgentError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = {
            let span_guard = this.span.as_ref().map(tracing::Span::enter);
            let inner_poll = this.inner.as_mut().poll_next(cx);
            drop(span_guard);
            inner_poll
        };

        match &poll {
            Poll::Ready(Some(Ok(StreamChunk::Token(text)))) => {
                if !text.is_empty() {
                    if !this.ttft_recorded {
                        this.ttft_recorded = true;
                        if let Some(ref span) = this.span {
                            record_ttft(span, this.started_at.elapsed());
                        }
                    }
                    this.completion_tokens = this
                        .completion_tokens
                        .saturating_add(estimate_token_count(text).max(1));
                }
            }
            Poll::Ready(Some(Ok(StreamChunk::Done))) | Poll::Ready(None) => {
                this.finish_span();
            }
            Poll::Ready(Some(Ok(StreamChunk::Cancelled))) | Poll::Ready(Some(Err(_))) => {
                if let Some(ref span) = this.span {
                    span.record("otel.status_code", "ERROR");
                }
                this.finish_span();
            }
            Poll::Pending => {}
        }

        poll
    }
}

/// Inject active W3C Trace Context (`traceparent` / `tracestate`) into outbound `reqwest` headers.
pub fn inject_w3c_trace_context(headers: &mut HeaderMap) {
    let context = tracing::Span::current().context();
    let mut injector = HeaderMapInjector(headers);
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut injector);
    });
}

struct HeaderMapInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderMapInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let (Ok(name), Ok(val)) = (
            HeaderName::from_bytes(key.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            self.0.insert(name, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::http::HeaderMap as AxumHeaderMap;
    use axum::routing::get;
    use axum::{Json, Router};
    use futures::StreamExt;
    use opentelemetry::Value;
    use opentelemetry_sdk::trace::InMemorySpanExporter;
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;
    use tracing::Subscriber;
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;
    use crate::agent::streaming::adapt_token_stream;
    use crate::client::http_client::FerrousApiClient;
    use crate::models::OrderDetails;

    fn with_in_memory_tracer<F, R>(f: F) -> (R, Vec<opentelemetry_sdk::trace::SpanData>)
    where
        F: FnOnce() -> R,
    {
        global::set_text_map_propagator(TraceContextPropagator::new());
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("ferrous-agent-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        let result = tracing::subscriber::with_default(subscriber, f);
        let _ = provider.force_flush();
        let spans = exporter
            .get_finished_spans()
            .expect("in-memory exporter should return spans");
        let _ = provider.shutdown();
        (result, spans)
    }

    fn build_in_memory_subscriber() -> (
        impl Subscriber + Send + Sync,
        SdkTracerProvider,
        InMemorySpanExporter,
    ) {
        global::set_text_map_propagator(TraceContextPropagator::new());
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let tracer = provider.tracer("ferrous-agent-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
        (subscriber, provider, exporter)
    }

    fn find_attr<'a>(span: &'a opentelemetry_sdk::trace::SpanData, key: &str) -> Option<&'a Value> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| &kv.value)
    }

    #[test]
    fn genai_chat_span_emits_semconv_attributes_with_in_memory_exporter() {
        let (_, spans) = with_in_memory_tracer(|| {
            let span = genai_chat_span("openai", "gpt-4o-mini", "chat");
            let _entered = span.enter();
            record_ttft(&span, Duration::from_micros(1_500));
            record_completion_tokens(&span, 42);
        });

        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "gen_ai.chat");
        assert_eq!(
            find_attr(span, GEN_AI_SYSTEM),
            Some(&Value::String("rig-core".into()))
        );
        assert_eq!(
            find_attr(span, GEN_AI_REQUEST_MODEL),
            Some(&Value::String("gpt-4o-mini".into()))
        );
        assert_eq!(
            find_attr(span, GEN_AI_PROVIDER_NAME),
            Some(&Value::String("openai".into()))
        );
        assert_eq!(
            find_attr(span, GEN_AI_OPERATION_NAME),
            Some(&Value::String("chat".into()))
        );
        assert_eq!(
            find_attr(span, GEN_AI_RESPONSE_COMPLETION_TOKENS),
            Some(&Value::I64(42))
        );
        assert_eq!(
            find_attr(span, GEN_AI_USAGE_OUTPUT_TOKENS),
            Some(&Value::I64(42))
        );
        assert_eq!(
            find_attr(span, GEN_AI_SERVER_TTFT_US),
            Some(&Value::I64(1_500))
        );
    }

    #[test]
    fn otel_sdk_disabled_initializes_cleanly_without_endpoint() {
        let mut disabled_env = HashMap::new();
        disabled_env.insert("OTEL_SDK_DISABLED".to_string(), "true".to_string());
        disabled_env.insert(
            "OTEL_EXPORTER_OTLP_ENDPOINT".to_string(),
            "http://127.0.0.1:4318".to_string(),
        );
        assert!(!is_otlp_enabled(&disabled_env));
        assert!(build_otlp_provider_from_map(&disabled_env).is_none());

        let empty_env = HashMap::new();
        assert!(!is_otlp_enabled(&empty_env));
        assert!(build_otlp_provider_from_map(&empty_env).is_none());

        let mut none_exporter_env = HashMap::new();
        none_exporter_env.insert("OTEL_TRACES_EXPORTER".to_string(), "none".to_string());
        none_exporter_env.insert(
            "OTEL_EXPORTER_OTLP_ENDPOINT".to_string(),
            "http://127.0.0.1:4318".to_string(),
        );
        assert!(!is_otlp_enabled(&none_exporter_env));
        assert!(build_otlp_provider_from_map(&none_exporter_env).is_none());

        let mut console_exporter_env = HashMap::new();
        console_exporter_env.insert("OTEL_TRACES_EXPORTER".to_string(), "console".to_string());
        console_exporter_env.insert(
            "OTEL_EXPORTER_OTLP_ENDPOINT".to_string(),
            "http://127.0.0.1:4318".to_string(),
        );
        assert!(!is_otlp_enabled(&console_exporter_env));
        assert!(build_otlp_provider_from_map(&console_exporter_env).is_none());

        assert_eq!(
            normalize_otlp_traces_endpoint("http://127.0.0.1:4318/"),
            "http://127.0.0.1:4318/v1/traces"
        );
        assert_eq!(
            normalize_otlp_traces_endpoint("http://127.0.0.1:4318/v1/traces/"),
            "http://127.0.0.1:4318/v1/traces"
        );
    }

    #[test]
    fn inject_w3c_trace_context_sets_valid_traceparent_header() {
        let (headers, spans) = with_in_memory_tracer(|| {
            let span = genai_chat_span("ollama", "qwen2.5:7b", "chat");
            let _entered = span.enter();
            let mut headers = HeaderMap::new();
            inject_w3c_trace_context(&mut headers);
            headers
        });

        assert_eq!(spans.len(), 1);
        let traceparent = headers
            .get("traceparent")
            .expect("traceparent header must be injected")
            .to_str()
            .expect("traceparent must be valid ASCII");
        let parts: Vec<&str> = traceparent.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent format: {traceparent}");
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1].len(), 32);
        assert!(parts[1].chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(parts[2].len(), 16);
        assert!(parts[2].chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(parts[3], "01");
        assert_eq!(parts[1], spans[0].span_context.trace_id().to_string());
    }

    #[tokio::test]
    async fn ferrous_api_client_propagates_w3c_traceparent_over_http() {
        let captured_traceparent: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let state = captured_traceparent.clone();

        let app =
            Router::new()
                .route(
                    "/api/v1/orders/:id",
                    get(
                        |State(captured): State<Arc<Mutex<Option<String>>>>,
                         headers: AxumHeaderMap| async move {
                            let tp = headers
                                .get("traceparent")
                                .and_then(|v| v.to_str().ok())
                                .map(ToOwned::to_owned);
                            *captured.lock().unwrap() = tp;
                            Json(OrderDetails {
                                order_id: "ord-1".to_string(),
                                customer_id: "cust-1".to_string(),
                                status: "processing".to_string(),
                                total_amount_cents: 1000,
                                currency: "USD".to_string(),
                                days_delayed: None,
                                freight_cents: None,
                                carrier: None,
                                items: vec![],
                                created_at: None,
                                updated_at: Some("2026-11-14T00:00:00Z".to_string()),
                            })
                        },
                    ),
                )
                .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let (subscriber, provider, exporter) = build_in_memory_subscriber();
        let _default_guard = tracing::subscriber::set_default(subscriber);

        let client = FerrousApiClient::new(format!("http://{addr}"), "test-token");
        let order = {
            let span = genai_chat_span("openai", "gpt-4o", "tool_call");
            let _entered = span.enter();
            // Future holds entered span context across await via Instrument
            use tracing::Instrument;
            client
                .get_order("ord-1")
                .instrument(span.clone())
                .await
                .expect("order request should succeed")
        };

        assert_eq!(order.order_id, "ord-1");
        let _ = provider.force_flush();
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);

        let received_tp = captured_traceparent
            .lock()
            .unwrap()
            .clone()
            .expect("server must receive traceparent header");
        let expected_trace_id = spans[0].span_context.trace_id().to_string();
        assert!(
            received_tp.starts_with(&format!("00-{expected_trace_id}-")),
            "expected traceparent with trace_id {expected_trace_id}, got {received_tp}"
        );

        let _ = provider.shutdown();
        server_handle.abort();
    }

    #[tokio::test]
    async fn instrument_chunk_stream_records_ttft_and_completion_tokens() {
        let (subscriber, provider, exporter) = build_in_memory_subscriber();
        let _default_guard = tracing::subscriber::set_default(subscriber);

        let span = genai_chat_span("ollama", "qwen2.5:7b", "stream");
        let started_at = Instant::now();
        let source = futures::stream::iter(vec![
            Ok(Some("Hello ".to_string())),
            Ok(Some("world!".to_string())),
        ]);
        let adapted = adapt_token_stream(source, CancellationToken::new());
        let instrumented = instrument_chunk_stream(adapted, span, started_at);

        let chunks: Vec<StreamChunk> = instrumented.map(|r| r.unwrap()).collect().await;
        assert_eq!(
            chunks,
            vec![
                StreamChunk::Token("Hello ".to_string()),
                StreamChunk::Token("world!".to_string()),
                StreamChunk::Done,
            ]
        );

        let _ = provider.force_flush();
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(
            find_attr(&spans[0], GEN_AI_RESPONSE_COMPLETION_TOKENS),
            Some(&Value::I64(2))
        );
        assert!(
            find_attr(&spans[0], GEN_AI_SERVER_TTFT_US).is_some(),
            "TTFT attribute must be recorded on first token"
        );
        let _ = provider.shutdown();
    }

    #[tokio::test]
    async fn otlp_http_exporter_posts_spans_to_v1_traces_endpoint() {
        use axum::body::Bytes;
        use axum::http::StatusCode;
        use axum::routing::post;

        let received_posts: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let state = received_posts.clone();

        let app = Router::new()
            .route(
                "/v1/traces",
                post(
                    |State(posts): State<Arc<Mutex<Vec<usize>>>>, body: Bytes| async move {
                        posts.lock().unwrap().push(body.len());
                        StatusCode::OK
                    },
                ),
            )
            .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let mut env = HashMap::new();
        env.insert(
            "OTEL_EXPORTER_OTLP_ENDPOINT".to_string(),
            format!("http://{addr}"),
        );
        let provider = build_otlp_provider_from_map(&env)
            .expect("OTLP provider should build with reqwest-blocking-client");
        let tracer = provider.tracer("ferrous-agent-otlp-test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        {
            let _default_guard = tracing::subscriber::set_default(subscriber);
            let span = genai_chat_span("openai", "gpt-4o-mini", "chat");
            let _entered = span.enter();
            record_completion_tokens(&span, 7);
        }

        tokio::task::spawn_blocking(move || {
            let _ = provider.force_flush();
            let _ = provider.shutdown();
        })
        .await
        .expect("provider shutdown task should complete");

        let posts = received_posts.lock().unwrap();
        assert!(
            !posts.is_empty(),
            "expected at least one OTLP POST to /v1/traces"
        );
        assert!(
            posts.iter().all(|len| *len > 0),
            "expected non-empty OTLP protobuf payload"
        );

        server_handle.abort();
    }

    #[tokio::test]
    async fn ferrous_api_client_search_qdrant_propagates_w3c_traceparent() {
        use axum::routing::post;
        use serde_json::json;
        use tracing::Instrument;

        let captured_traceparent: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let state = captured_traceparent.clone();

        let app =
            Router::new()
                .route(
                    "/collections/:collection/points/search",
                    post(
                        |State(captured): State<Arc<Mutex<Option<String>>>>,
                         headers: AxumHeaderMap| async move {
                            let tp = headers
                                .get("traceparent")
                                .and_then(|v| v.to_str().ok())
                                .map(ToOwned::to_owned);
                            *captured.lock().unwrap() = tp;
                            Json(json!({ "result": [] }))
                        },
                    ),
                )
                .with_state(state);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let (subscriber, provider, exporter) = build_in_memory_subscriber();
        let _default_guard = tracing::subscriber::set_default(subscriber);

        let client = FerrousApiClient::new(format!("http://{addr}"), "test-token");
        let hits = {
            let span = genai_chat_span("openai", "gpt-4o", "tool_call");
            client
                .search_qdrant("kb", "delayed shipment", None, 3)
                .instrument(span)
                .await
                .expect("search_qdrant request should succeed")
        };
        assert!(hits.is_empty());

        let _ = provider.force_flush();
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);

        let received_tp = captured_traceparent
            .lock()
            .unwrap()
            .clone()
            .expect("Qdrant server must receive traceparent header");
        let expected_trace_id = spans[0].span_context.trace_id().to_string();
        assert!(
            received_tp.starts_with(&format!("00-{expected_trace_id}-")),
            "expected traceparent with trace_id {expected_trace_id}, got {received_tp}"
        );

        let _ = provider.shutdown();
        server_handle.abort();
    }

    #[tokio::test]
    async fn instrument_chunk_stream_skips_empty_tokens_and_records_errors_and_drop() {
        use opentelemetry::trace::Status;

        // 1. Empty Token("") does not trigger TTFT or token count; Cancelled sets ERROR status.
        {
            let (subscriber, provider, exporter) = build_in_memory_subscriber();
            let _default_guard = tracing::subscriber::set_default(subscriber);

            let span = genai_chat_span("ollama", "qwen2.5:7b", "stream");
            let source: BoxedChunkStream = Box::pin(futures::stream::iter(vec![
                Ok(StreamChunk::Token(String::new())),
                Ok(StreamChunk::Cancelled),
            ]));
            let mut instrumented = instrument_chunk_stream(source, span, Instant::now());
            let first = instrumented.next().await.unwrap().unwrap();
            assert_eq!(first, StreamChunk::Token(String::new()));
            let second = instrumented.next().await.unwrap().unwrap();
            assert_eq!(second, StreamChunk::Cancelled);

            let _ = provider.force_flush();
            let spans = exporter.get_finished_spans().unwrap();
            assert_eq!(spans.len(), 1);
            assert!(
                find_attr(&spans[0], GEN_AI_SERVER_TTFT_US).is_none(),
                "empty token must not record TTFT"
            );
            assert_eq!(
                find_attr(&spans[0], GEN_AI_RESPONSE_COMPLETION_TOKENS),
                Some(&Value::I64(0))
            );
            assert!(
                matches!(spans[0].status, Status::Error { .. }),
                "Cancelled chunk must set Error status, got {:?}",
                spans[0].status
            );
            let _ = provider.shutdown();
        }

        // 2. Mid-stream Err(_) records tokens received so far and sets ERROR status.
        {
            let (subscriber, provider, exporter) = build_in_memory_subscriber();
            let _default_guard = tracing::subscriber::set_default(subscriber);

            let span = genai_chat_span("ollama", "qwen2.5:7b", "stream");
            let source: BoxedChunkStream = Box::pin(futures::stream::iter(vec![
                Ok(StreamChunk::Token("partial".to_string())),
                Err(AgentError::Stream("upstream reset".to_string())),
            ]));
            let mut instrumented = instrument_chunk_stream(source, span, Instant::now());
            assert!(matches!(
                instrumented.next().await,
                Some(Ok(StreamChunk::Token(_)))
            ));
            assert!(matches!(instrumented.next().await, Some(Err(_))));

            let _ = provider.force_flush();
            let spans = exporter.get_finished_spans().unwrap();
            assert_eq!(spans.len(), 1);
            assert_eq!(
                find_attr(&spans[0], GEN_AI_RESPONSE_COMPLETION_TOKENS),
                Some(&Value::I64(1))
            );
            assert!(find_attr(&spans[0], GEN_AI_SERVER_TTFT_US).is_some());
            assert!(matches!(spans[0].status, Status::Error { .. }));
            let _ = provider.shutdown();
        }

        // 3. Early Drop finishes the span and records tokens received before drop.
        {
            let (subscriber, provider, exporter) = build_in_memory_subscriber();
            let _default_guard = tracing::subscriber::set_default(subscriber);

            let span = genai_chat_span("ollama", "qwen2.5:7b", "stream");
            let source: BoxedChunkStream = Box::pin(futures::stream::iter(vec![
                Ok(StreamChunk::Token("one two".to_string())),
                Ok(StreamChunk::Token("three".to_string())),
            ]));
            let mut instrumented = instrument_chunk_stream(source, span, Instant::now());
            assert!(matches!(
                instrumented.next().await,
                Some(Ok(StreamChunk::Token(_)))
            ));
            drop(instrumented);

            let _ = provider.force_flush();
            let spans = exporter.get_finished_spans().unwrap();
            assert_eq!(spans.len(), 1);
            assert_eq!(
                find_attr(&spans[0], GEN_AI_RESPONSE_COMPLETION_TOKENS),
                Some(&Value::I64(2))
            );
            assert!(find_attr(&spans[0], GEN_AI_SERVER_TTFT_US).is_some());
            let _ = provider.shutdown();
        }
    }
}
