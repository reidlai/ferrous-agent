//! Axum Server-Sent Events (SSE) hub and `GET /api/v1/benchmarks/{run_id}/stream` router.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use axum::Router;
use futures::Stream;
use tokio::sync::{broadcast, RwLock};

use crate::benchmark::runner::{BenchmarkEvent, DEFAULT_EVENT_CHANNEL_CAPACITY};

#[derive(Clone, Debug)]
struct RunChannel {
    tx: broadcast::Sender<BenchmarkEvent>,
    terminal: Arc<RwLock<Option<BenchmarkEvent>>>,
}

/// Thread-safe registry mapping benchmark `run_id`s to live broadcast channels and terminal state.
#[derive(Clone, Debug, Default)]
pub struct BenchmarkSseHub {
    runs: Arc<RwLock<HashMap<String, RunChannel>>>,
}

impl BenchmarkSseHub {
    /// Create an empty [`BenchmarkSseHub`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new benchmark run and return its [`broadcast::Sender<BenchmarkEvent>`].
    pub async fn register_run(
        &self,
        run_id: impl Into<String>,
    ) -> broadcast::Sender<BenchmarkEvent> {
        let (tx, _rx) = broadcast::channel(DEFAULT_EVENT_CHANNEL_CAPACITY);
        self.register_sender(run_id, tx.clone()).await;
        tx
    }

    /// Register an existing broadcast sender for `run_id` and track its terminal completion event.
    pub async fn register_sender(
        &self,
        run_id: impl Into<String>,
        sender: broadcast::Sender<BenchmarkEvent>,
    ) {
        let terminal = Arc::new(RwLock::new(None));
        let terminal_clone = Arc::clone(&terminal);
        let mut rx = sender.subscribe();

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) if matches!(ev, BenchmarkEvent::BenchmarkCompleted { .. }) => {
                        *terminal_clone.write().await = Some(ev);
                        break;
                    }
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        self.runs.write().await.insert(
            run_id.into(),
            RunChannel {
                tx: sender,
                terminal,
            },
        );
    }

    /// Subscribe to live events for `run_id`, or `None` if `run_id` is unknown.
    pub async fn subscribe(&self, run_id: &str) -> Option<broadcast::Receiver<BenchmarkEvent>> {
        self.runs
            .read()
            .await
            .get(run_id)
            .map(|ch| ch.tx.subscribe())
    }

    async fn subscribe_with_terminal(
        &self,
        run_id: &str,
    ) -> Option<(broadcast::Receiver<BenchmarkEvent>, Option<BenchmarkEvent>)> {
        let channel = self.runs.read().await.get(run_id).cloned()?;
        let rx = channel.tx.subscribe();
        let terminal = channel.terminal.read().await.clone();
        Some((rx, terminal))
    }

    /// Return the current number of active subscribers for `run_id`.
    pub async fn subscriber_count(&self, run_id: &str) -> usize {
        self.runs
            .read()
            .await
            .get(run_id)
            .map_or(0, |ch| ch.tx.receiver_count())
    }

    /// Remove a completed run from the hub.
    pub async fn remove_run(&self, run_id: &str) {
        self.runs.write().await.remove(run_id);
    }
}

/// Build the Axum [`Router`] exposing `GET /api/v1/benchmarks/:run_id/stream`.
pub fn benchmark_sse_router(hub: BenchmarkSseHub) -> Router {
    Router::new()
        .route(
            "/api/v1/benchmarks/:run_id/stream",
            get(stream_benchmark_events),
        )
        .with_state(hub)
}

fn encode_sse_event(ev: &BenchmarkEvent) -> Event {
    let event_name = ev.event_name();
    match Event::default().event(event_name).json_data(ev) {
        Ok(event) => event,
        Err(err) => Event::default()
            .event("error")
            .data(format!("serialization error: {err}")),
    }
}

async fn stream_benchmark_events(
    Path(run_id): Path<String>,
    State(hub): State<BenchmarkSseHub>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let (rx, terminal) = hub
        .subscribe_with_terminal(&run_id)
        .await
        .ok_or(StatusCode::NOT_FOUND)?;

    let stream = futures::stream::unfold(
        (rx, terminal, false),
        |(mut rx, mut terminal, finished)| async move {
            if finished {
                return None;
            }
            if let Some(ev) = terminal.take() {
                return Some((Ok(encode_sse_event(&ev)), (rx, None, true)));
            }
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        let is_terminal = matches!(ev, BenchmarkEvent::BenchmarkCompleted { .. });
                        return Some((Ok(encode_sse_event(&ev)), (rx, None, is_terminal)));
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        },
    );

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    ))
}

/// Serve the benchmark SSE router on a pre-bound `TcpListener`.
pub async fn serve_sse(
    listener: tokio::net::TcpListener,
    hub: BenchmarkSseHub,
) -> std::io::Result<()> {
    let app = benchmark_sse_router(hub);
    axum::serve(listener, app).await
}

#[cfg(test)]
mod tests {
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::benchmark::cases::BenchmarkSuite;
    use crate::benchmark::runner::{BenchmarkRunner, DryRunDispatch};

    #[tokio::test]
    async fn unknown_run_id_returns_404_not_found() {
        let hub = BenchmarkSseHub::new();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let _ = serve_sse(listener, hub).await;
        });

        let resp = reqwest::get(format!(
            "http://{addr}/api/v1/benchmarks/does-not-exist/stream"
        ))
        .await
        .expect("HTTP request should complete");
        assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);

        server.abort();
    }

    #[tokio::test]
    async fn streams_token_chunk_case_completed_and_benchmark_completed_sse_events() {
        let hub = BenchmarkSseHub::new();
        let run_id = "run-sse-test-1";
        let tx = hub.register_run(run_id).await;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hub_clone = hub.clone();
        let server = tokio::spawn(async move {
            let _ = serve_sse(listener, hub_clone).await;
        });

        let client_url = format!("http://{addr}/api/v1/benchmarks/{run_id}/stream");
        let resp = reqwest::get(&client_url).await.expect("SSE connect");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let body_task = tokio::spawn(async move { resp.text().await.expect("SSE body") });

        let dispatch = Arc::new(DryRunDispatch::default().with_chunk_delay(Duration::ZERO));
        let runner = BenchmarkRunner::with_broadcaster(dispatch, tx)
            .unwrap()
            .with_dry_run(true);
        let report = runner
            .run_suite_with_id(run_id, BenchmarkSuite::All, CancellationToken::new())
            .await;
        assert_eq!(report.failed_cases, 0);

        let body = tokio::time::timeout(Duration::from_secs(5), body_task)
            .await
            .expect("SSE stream should finish")
            .expect("client task should succeed");

        assert!(
            body.contains("event: token_chunk"),
            "expected token_chunk SSE event in:\n{body}"
        );
        assert!(
            body.contains("event: case_completed"),
            "expected case_completed SSE event in:\n{body}"
        );
        assert!(
            body.contains("event: benchmark_completed"),
            "expected benchmark_completed SSE event in:\n{body}"
        );

        server.abort();
    }

    #[tokio::test]
    async fn late_sse_subscriber_receives_terminal_event_and_closes() {
        let hub = BenchmarkSseHub::new();
        let run_id = "run-sse-late-1";
        let tx = hub.register_run(run_id).await;

        let dispatch = Arc::new(DryRunDispatch::default().with_chunk_delay(Duration::ZERO));
        let runner = BenchmarkRunner::with_broadcaster(dispatch, tx)
            .unwrap()
            .with_dry_run(true);
        let report = runner
            .run_suite_with_id(run_id, BenchmarkSuite::Tools, CancellationToken::new())
            .await;
        assert_eq!(report.failed_cases, 0);

        // Allow the background watcher task to record the terminal event.
        tokio::task::yield_now().await;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = serve_sse(listener, hub).await;
        });

        let client_url = format!("http://{addr}/api/v1/benchmarks/{run_id}/stream");
        let resp = tokio::time::timeout(Duration::from_secs(5), reqwest::get(&client_url))
            .await
            .expect("late SSE connect should not hang")
            .expect("HTTP request should succeed");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);

        let body = tokio::time::timeout(Duration::from_secs(5), resp.text())
            .await
            .expect("late SSE stream should close cleanly")
            .expect("SSE body");
        assert!(
            body.contains("event: benchmark_completed"),
            "expected terminal benchmark_completed event in:\n{body}"
        );

        server.abort();
    }
}
