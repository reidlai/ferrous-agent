use std::pin::Pin;

use futures::{Stream, StreamExt};
use rig_core::operation::Completion;
use rig_core::streaming::{Item, StreamEvent, Streamed};
use tokio_util::sync::CancellationToken;

use crate::agent::AgentError;

/// Unified streaming chunk yielded by `CopilotAgent` and `LlmClient`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    /// Incremental assistant text delta.
    Token(String),
    /// Provider stream completed normally.
    Done,
    /// Stream was halted via `CancellationToken`.
    Cancelled,
}

/// Boxed, `Send` stream of `StreamChunk` results.
pub type BoxedChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, AgentError>> + Send>>;

/// Adapt a `rig-core` completion stream into a cancellation-aware `BoxedChunkStream`.
pub fn adapt_rig_stream(
    stream: Streamed<Completion>,
    cancel: CancellationToken,
) -> BoxedChunkStream {
    let mapped = stream.map(|item| match item {
        Ok(Item::Event(StreamEvent::Text { text, .. })) => Ok(Some(text)),
        Ok(_) => Ok(None),
        Err(err) => Err(AgentError::from(err)),
    });
    adapt_token_stream(mapped, cancel)
}

/// Adapt any fallible text-delta stream into a cancellation-aware `BoxedChunkStream`.
///
/// When `cancel` is triggered, the inner stream is dropped immediately, `StreamChunk::Cancelled`
/// is yielded once, and the stream terminates.
pub fn adapt_token_stream<S>(stream: S, cancel: CancellationToken) -> BoxedChunkStream
where
    S: Stream<Item = Result<Option<String>, AgentError>> + Send + 'static,
{
    let initial = Some((Box::pin(stream), cancel));
    Box::pin(futures::stream::unfold(initial, |state| async move {
        let (mut inner, cancel) = state?;
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    return Some((Ok(StreamChunk::Cancelled), None));
                }
                next_item = inner.next() => {
                    match next_item {
                        Some(Ok(Some(token))) if !token.is_empty() => {
                            return Some((Ok(StreamChunk::Token(token)), Some((inner, cancel))));
                        }
                        Some(Ok(_)) => continue,
                        Some(Err(err)) => {
                            return Some((Err(err), None));
                        }
                        None => {
                            return Some((Ok(StreamChunk::Done), None));
                        }
                    }
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn streams_tokens_and_terminates_with_done() {
        let cancel = CancellationToken::new();
        let source = futures::stream::iter(vec![
            Ok(Some("Hello".to_string())),
            Ok(None),
            Ok(Some(String::new())),
            Ok(Some(", world!".to_string())),
        ]);

        let chunks: Vec<StreamChunk> = adapt_token_stream(source, cancel)
            .map(|res| res.expect("chunk should be ok"))
            .collect()
            .await;

        assert_eq!(
            chunks,
            vec![
                StreamChunk::Token("Hello".to_string()),
                StreamChunk::Token(", world!".to_string()),
                StreamChunk::Done,
            ]
        );
    }

    #[tokio::test]
    async fn cancellation_halts_pending_stream_promptly() {
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::channel::<Result<Option<String>, AgentError>>(4);

        tx.send(Ok(Some("first ".to_string()))).await.unwrap();

        let source = async_stream_from_rx(rx);
        let mut stream = adapt_token_stream(source, cancel.clone());

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, StreamChunk::Token("first ".to_string()));

        // Cancel while the inner stream is waiting for the next token.
        cancel.cancel();

        let second = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await
            .expect("cancellation should resolve immediately")
            .expect("stream should yield Cancelled")
            .expect("Cancelled should be Ok");
        assert_eq!(second, StreamChunk::Cancelled);

        // Stream must terminate cleanly after Cancelled.
        assert!(stream.next().await.is_none());

        // Sending after cancellation should fail because the receiver was dropped.
        assert!(tx.send(Ok(Some("never".to_string()))).await.is_err());
    }

    #[tokio::test]
    async fn pre_cancelled_token_yields_cancelled_immediately() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let source = futures::stream::iter(vec![Ok(Some("ignored".to_string()))]);
        let chunks: Vec<StreamChunk> = adapt_token_stream(source, cancel)
            .map(|res| res.expect("chunk should be ok"))
            .collect()
            .await;

        assert_eq!(chunks, vec![StreamChunk::Cancelled]);
    }

    #[tokio::test]
    async fn error_is_yielded_and_stream_terminates_cleanly() {
        let cancel = CancellationToken::new();
        let source = futures::stream::iter(vec![
            Ok(Some("ok-token".to_string())),
            Err(AgentError::Stream("boom".to_string())),
            Ok(Some("never-reached".to_string())),
        ]);

        let mut stream = adapt_token_stream(source, cancel);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, StreamChunk::Token("ok-token".to_string()));

        let second = stream.next().await.unwrap();
        assert!(matches!(second, Err(AgentError::Stream(msg)) if msg == "boom"));

        // Stream must terminate cleanly after yielding an error.
        assert!(stream.next().await.is_none());
    }

    fn async_stream_from_rx(
        rx: mpsc::Receiver<Result<Option<String>, AgentError>>,
    ) -> impl Stream<Item = Result<Option<String>, AgentError>> + Send + 'static {
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
    }
}
