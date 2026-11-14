//! Microsecond-precision streaming latency and throughput metrics.

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::agent::streaming::{BoxedChunkStream, StreamChunk};
use crate::agent::AgentError;

/// Microsecond-precision latency and throughput metrics for a single benchmark stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Time to first token in microseconds.
    pub ttft_us: u64,
    /// Mean inter-token latency in microseconds.
    pub itl_mean_us: f64,
    /// Inter-token latency jitter (standard deviation) in microseconds.
    pub itl_jitter_us: f64,
    /// Throughput in tokens (stream chunks) per second.
    pub tps: f64,
    /// Total non-empty token chunks observed.
    pub token_count: usize,
    /// Total stream duration in microseconds.
    pub total_duration_us: u64,
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            ttft_us: 0,
            itl_mean_us: 0.0,
            itl_jitter_us: 0.0,
            tps: 0.0,
            token_count: 0,
            total_duration_us: 0,
        }
    }
}

/// Collector that records `tokio::time::Instant` timestamps for each incoming `StreamChunk::Token`.
#[derive(Debug, Clone)]
pub struct StreamTimingCollector {
    start: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    itl_samples_us: Vec<u64>,
    token_count: usize,
}

impl StreamTimingCollector {
    /// Create a new collector anchored at `Instant::now()`.
    pub fn new() -> Self {
        Self::with_start(Instant::now())
    }

    /// Create a new collector anchored at an explicit start timestamp.
    pub fn with_start(start: Instant) -> Self {
        Self {
            start,
            first_token_at: None,
            last_token_at: None,
            itl_samples_us: Vec::new(),
            token_count: 0,
        }
    }

    /// Record arrival of a token at `Instant::now()`.
    pub fn record_token(&mut self) {
        self.record_token_at(Instant::now());
    }

    /// Record arrival of a token at an explicit timestamp.
    pub fn record_token_at(&mut self, now: Instant) {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
        }
        if let Some(prev) = self.last_token_at {
            let delta_us = now.saturating_duration_since(prev).as_micros() as u64;
            self.itl_samples_us.push(delta_us);
        }
        self.last_token_at = Some(now);
        self.token_count += 1;
    }

    /// Finalize metrics using `Instant::now()`.
    pub fn finish(&self) -> BenchmarkMetrics {
        self.finish_at(Instant::now())
    }

    /// Finalize metrics using an explicit end timestamp.
    pub fn finish_at(&self, end: Instant) -> BenchmarkMetrics {
        let total_duration_us = end.saturating_duration_since(self.start).as_micros() as u64;

        if self.token_count == 0 {
            return BenchmarkMetrics {
                ttft_us: 0,
                itl_mean_us: 0.0,
                itl_jitter_us: 0.0,
                tps: 0.0,
                token_count: 0,
                total_duration_us,
            };
        }

        let ttft_us = self
            .first_token_at
            .map(|first| first.saturating_duration_since(self.start).as_micros() as u64)
            .unwrap_or(0);

        let (itl_mean_us, itl_jitter_us) = if self.itl_samples_us.is_empty() {
            (0.0, 0.0)
        } else {
            let count = self.itl_samples_us.len() as f64;
            let sum: f64 = self.itl_samples_us.iter().map(|&v| v as f64).sum();
            let mean = sum / count;
            let variance: f64 = self
                .itl_samples_us
                .iter()
                .map(|&v| {
                    let diff = (v as f64) - mean;
                    diff * diff
                })
                .sum::<f64>()
                / count;
            (mean, variance.sqrt())
        };

        let tps = if total_duration_us == 0 {
            0.0
        } else {
            (self.token_count as f64) / ((total_duration_us as f64) / 1_000_000.0)
        };

        BenchmarkMetrics {
            ttft_us,
            itl_mean_us,
            itl_jitter_us,
            tps,
            token_count: self.token_count,
            total_duration_us,
        }
    }
}

impl Default for StreamTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Consume a [`BoxedChunkStream`] and compute [`BenchmarkMetrics`] with microsecond precision.
pub async fn collect_stream_metrics<F>(
    mut stream: BoxedChunkStream,
    mut on_token: F,
) -> Result<(String, BenchmarkMetrics), AgentError>
where
    F: FnMut(&str) + Send,
{
    let mut collector = StreamTimingCollector::new();
    let mut output = String::new();

    while let Some(item) = stream.next().await {
        match item? {
            StreamChunk::Token(text) => {
                if !text.is_empty() {
                    collector.record_token();
                    on_token(&text);
                    output.push_str(&text);
                }
            }
            StreamChunk::Done => break,
            StreamChunk::Cancelled => {
                return Err(AgentError::Stream(
                    "benchmark stream was cancelled".to_string(),
                ));
            }
        }
    }

    Ok((output, collector.finish()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::agent::streaming::adapt_token_stream;

    #[test]
    fn measures_ttft_itl_jitter_and_tps_with_microsecond_precision() {
        let start = Instant::now();
        let mut collector = StreamTimingCollector::with_start(start);

        // 50ms TTFT
        let t1 = start + Duration::from_millis(50);
        collector.record_token_at(t1);

        // ITL #1 = 10ms (10_000 us)
        let t2 = t1 + Duration::from_millis(10);
        collector.record_token_at(t2);

        // ITL #2 = 30ms (30_000 us) -> mean = 20_000 us, std-dev = 10_000 us
        let t3 = t2 + Duration::from_millis(30);
        collector.record_token_at(t3);

        // Finalize 10ms after last token -> total = 100ms (100_000 us), tps = 3 / 0.1s = 30.0
        let end = t3 + Duration::from_millis(10);
        let metrics = collector.finish_at(end);

        assert_eq!(metrics.ttft_us, 50_000);
        assert_eq!(metrics.token_count, 3);
        assert_eq!(metrics.total_duration_us, 100_000);
        assert!((metrics.itl_mean_us - 20_000.0).abs() < 1e-6);
        assert!((metrics.itl_jitter_us - 10_000.0).abs() < 1e-6);
        assert!((metrics.tps - 30.0).abs() < 1e-6);
    }

    #[test]
    fn empty_stream_returns_zero_without_nan() {
        let start = Instant::now();
        let collector = StreamTimingCollector::with_start(start);
        let metrics = collector.finish_at(start + Duration::from_millis(25));

        assert_eq!(metrics.token_count, 0);
        assert_eq!(metrics.ttft_us, 0);
        assert_eq!(metrics.tps, 0.0);
        assert_eq!(metrics.itl_mean_us, 0.0);
        assert_eq!(metrics.itl_jitter_us, 0.0);
        assert!(!metrics.tps.is_nan());
        assert!(!metrics.itl_jitter_us.is_nan());
        assert!(!metrics.itl_mean_us.is_nan());
    }

    #[test]
    fn single_token_stream_has_zero_itl_and_no_nan() {
        let start = Instant::now();
        let mut collector = StreamTimingCollector::with_start(start);
        let t1 = start + Duration::from_millis(20);
        collector.record_token_at(t1);
        let metrics = collector.finish_at(t1 + Duration::from_millis(5));

        assert_eq!(metrics.token_count, 1);
        assert_eq!(metrics.ttft_us, 20_000);
        assert_eq!(metrics.itl_mean_us, 0.0);
        assert_eq!(metrics.itl_jitter_us, 0.0);
        assert!((metrics.tps - 40.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn collect_stream_metrics_consumes_boxed_chunk_stream() {
        let (tx, rx) = mpsc::channel::<Result<Option<String>, AgentError>>(8);
        let source = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });
        let stream = adapt_token_stream(source, CancellationToken::new());

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            tx.send(Ok(Some("Hello ".to_string()))).await.unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
            tx.send(Ok(Some("world".to_string()))).await.unwrap();
        });

        let mut observed = Vec::new();
        let (text, metrics) = collect_stream_metrics(stream, |tok| {
            observed.push(tok.to_string());
        })
        .await
        .expect("stream collection should succeed");

        assert_eq!(text, "Hello world");
        assert_eq!(observed, vec!["Hello ", "world"]);
        assert_eq!(metrics.token_count, 2);
        assert!(metrics.ttft_us > 0);
        assert!(metrics.itl_mean_us > 0.0);
        assert_eq!(metrics.itl_jitter_us, 0.0);
        assert!(metrics.tps > 0.0);
    }

    #[tokio::test]
    async fn collect_stream_metrics_returns_error_on_cancelled_chunk() {
        let cancel = CancellationToken::new();
        let source = futures::stream::iter(vec![
            Ok(Some("partial ".to_string())),
            Ok(Some("ignored".to_string())),
        ]);
        let mut stream = adapt_token_stream(source, cancel.clone());

        // Read first token, then cancel before consuming the rest.
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, StreamChunk::Token("partial ".to_string()));
        cancel.cancel();

        let err = collect_stream_metrics(stream, |_| {})
            .await
            .expect_err("cancelled stream must return Err");
        assert!(matches!(err, AgentError::Stream(ref msg) if msg.contains("cancelled")));
    }
}
