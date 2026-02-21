use async_trait::async_trait;
use futures_util::stream::Stream;
use std::pin::Pin;

// Boundary types owned by channels — re-imported here for use in voice trait signatures.
pub use crate::channels::traits::{AudioFormat, AudioOutput, VoiceAttachment, VoiceOutput};

/// Result of speech-to-text transcription.
#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub confidence: Option<f32>,
}

/// Configuration for voice synthesis.
#[derive(Debug, Clone)]
pub struct VoiceProfile {
    /// Provider-specific voice/character identifier.
    pub voice_id: Option<String>,
    /// Target audio format for the output.
    pub output_format: AudioFormat,
    /// Speech rate multiplier (1.0 = normal).
    pub speed: Option<f32>,
}

/// Speech-to-text provider trait.
#[async_trait]
pub trait SttProvider: Send + Sync {
    /// Human-readable provider name.
    fn name(&self) -> &str;

    /// Transcribe audio bytes into text.
    async fn transcribe(&self, audio: &[u8], format: AudioFormat) -> anyhow::Result<Transcript>;

    /// Check if the provider API is reachable.
    async fn health_check(&self) -> bool;
}

/// Text-to-speech provider trait.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Human-readable provider name.
    fn name(&self) -> &str;

    /// Synthesize text into audio.
    async fn synthesize(&self, text: &str, profile: &VoiceProfile) -> anyhow::Result<AudioOutput>;

    /// Check if the provider API is reachable.
    async fn health_check(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Streaming provider traits
// ---------------------------------------------------------------------------

/// Incremental speech-to-text result from a streaming transcription.
#[derive(Debug, Clone)]
pub struct TranscriptChunk {
    pub text: String,
    /// Whether this chunk represents a finalized segment.
    pub is_final: bool,
}

/// Incremental audio chunk from a streaming synthesis.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub format: AudioFormat,
    /// Monotonically increasing sequence number within a stream.
    pub sequence: u32,
}

/// Streaming speech-to-text provider trait.
#[async_trait]
pub trait StreamingSttProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Accept a stream of raw audio bytes and return a stream of transcript chunks.
    async fn transcribe_stream(
        &self,
        audio: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>,
        format: AudioFormat,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<TranscriptChunk>> + Send>>>;
}

/// Streaming text-to-speech provider trait.
#[async_trait]
pub trait StreamingTtsProvider: Send + Sync {
    fn name(&self) -> &str;

    /// Synthesize text into a stream of audio chunks.
    async fn synthesize_stream(
        &self,
        text: &str,
        profile: &VoiceProfile,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<AudioChunk>> + Send>>>;
}

// ---------------------------------------------------------------------------
// Voice pipeline trait (high-level orchestration)
// ---------------------------------------------------------------------------

/// How the pipeline decides whether to reply with voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyMode {
    /// Always reply with a voice message.
    VoiceOnly,
    /// Reply with both text and voice.
    TextAndVoice,
    /// Mirror the input: voice replies to voice, text to text.
    MatchInput,
}

/// High-level voice pipeline trait that channels consume.
///
/// Provides both batch and streaming interfaces. Batch methods are required;
/// streaming methods default to unsupported so implementations can opt in.
#[async_trait]
pub trait VoicePipeline: Send + Sync {
    /// Human-readable pipeline name.
    fn name(&self) -> &str;

    // -- Batch mode (required) --

    /// Transcribe an incoming voice attachment to text.
    async fn transcribe_incoming(&self, attachment: &VoiceAttachment) -> anyhow::Result<String>;

    /// Synthesize an outgoing response, returning voice or text based on pipeline policy.
    async fn synthesize_response(
        &self,
        text: &str,
        was_voice_input: bool,
    ) -> anyhow::Result<VoiceOutput>;

    // -- Streaming mode (optional) --

    /// Whether this pipeline supports streaming I/O.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Stream-transcribe incoming audio chunks to transcript chunks.
    async fn transcribe_stream(
        &self,
        _audio: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>,
        _format: AudioFormat,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<TranscriptChunk>> + Send>>> {
        anyhow::bail!("Streaming transcription not supported by {}", self.name())
    }

    /// Stream-synthesize a response into audio chunks.
    async fn synthesize_stream(
        &self,
        _text: &str,
        _was_voice_input: bool,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = anyhow::Result<AudioChunk>> + Send>>> {
        anyhow::bail!("Streaming synthesis not supported by {}", self.name())
    }

    /// Health check for both STT and TTS providers.
    async fn health_check(&self) -> (bool, bool);
}

