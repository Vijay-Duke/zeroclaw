use async_trait::async_trait;

use crate::channels::traits::{VoiceAttachment, VoiceOutput};
use super::traits::{ReplyMode, SttProvider, TtsProvider, VoicePipeline, VoiceProfile};

/// Default voice pipeline: batch-mode STT/TTS using file download/upload.
///
/// This is the standard pipeline for channels that exchange voice messages as
/// complete files (e.g. Telegram, Discord). Channels that need streaming audio
/// (WebSocket, phone) should use a streaming pipeline implementation instead.
pub struct DefaultVoicePipeline {
    stt: Box<dyn SttProvider>,
    tts: Box<dyn TtsProvider>,
    voice_profile: VoiceProfile,
    reply_mode: ReplyMode,
    http_client: reqwest::Client,
}

impl DefaultVoicePipeline {
    pub fn new(
        stt: Box<dyn SttProvider>,
        tts: Box<dyn TtsProvider>,
        voice_profile: VoiceProfile,
        reply_mode: ReplyMode,
    ) -> Self {
        Self {
            stt,
            tts,
            voice_profile,
            reply_mode,
            http_client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl VoicePipeline for DefaultVoicePipeline {
    fn name(&self) -> &str {
        "default"
    }

    async fn transcribe_incoming(
        &self,
        attachment: &VoiceAttachment,
    ) -> anyhow::Result<String> {
        let audio_bytes = self
            .http_client
            .get(&attachment.url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;

        let transcript = self.stt.transcribe(&audio_bytes, attachment.format).await?;

        if transcript.text.trim().is_empty() {
            anyhow::bail!("Voice message transcription produced empty text");
        }

        tracing::info!(
            "Voice STT ({}) transcribed {} bytes \u{2192} {} chars",
            self.stt.name(),
            audio_bytes.len(),
            transcript.text.len()
        );

        Ok(transcript.text)
    }

    async fn synthesize_response(
        &self,
        response_text: &str,
        was_voice_input: bool,
    ) -> anyhow::Result<VoiceOutput> {
        let should_voice = match self.reply_mode {
            ReplyMode::VoiceOnly | ReplyMode::TextAndVoice => true,
            ReplyMode::MatchInput => was_voice_input,
        };

        if !should_voice {
            return Ok(VoiceOutput::TextOnly(response_text.to_string()));
        }

        let audio = self
            .tts
            .synthesize(response_text, &self.voice_profile)
            .await?;

        tracing::info!(
            "Voice TTS ({}) synthesized {} chars \u{2192} {} bytes",
            self.tts.name(),
            response_text.len(),
            audio.data.len()
        );

        Ok(VoiceOutput::WithAudio {
            text: response_text.to_string(),
            audio,
        })
    }

    async fn health_check(&self) -> (bool, bool) {
        let (stt_ok, tts_ok) = tokio::join!(self.stt.health_check(), self.tts.health_check());
        (stt_ok, tts_ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::traits::{AudioFormat, AudioOutput};
    use crate::voice::traits::Transcript;
    use async_trait::async_trait;

    struct MockStt;

    #[async_trait]
    impl SttProvider for MockStt {
        fn name(&self) -> &str {
            "mock-stt"
        }
        async fn transcribe(
            &self,
            _audio: &[u8],
            _format: AudioFormat,
        ) -> anyhow::Result<Transcript> {
            Ok(Transcript {
                text: "hello world".into(),
                language: Some("en".into()),
                confidence: Some(0.95),
            })
        }
        async fn health_check(&self) -> bool {
            true
        }
    }

    struct MockTts;

    #[async_trait]
    impl TtsProvider for MockTts {
        fn name(&self) -> &str {
            "mock-tts"
        }
        async fn synthesize(
            &self,
            text: &str,
            _profile: &VoiceProfile,
        ) -> anyhow::Result<AudioOutput> {
            Ok(AudioOutput {
                data: text.as_bytes().to_vec(),
                format: AudioFormat::OggOpus,
                duration_ms: Some(1000),
            })
        }
        async fn health_check(&self) -> bool {
            true
        }
    }

    fn make_pipeline(reply_mode: ReplyMode) -> DefaultVoicePipeline {
        DefaultVoicePipeline::new(
            Box::new(MockStt),
            Box::new(MockTts),
            VoiceProfile {
                voice_id: None,
                output_format: AudioFormat::OggOpus,
                speed: None,
            },
            reply_mode,
        )
    }

    #[tokio::test]
    async fn synthesize_response_match_input_voice() {
        let pipeline = make_pipeline(ReplyMode::MatchInput);
        let result = pipeline
            .synthesize_response("test response", true)
            .await
            .unwrap();
        assert!(matches!(result, VoiceOutput::WithAudio { .. }));
    }

    #[tokio::test]
    async fn synthesize_response_match_input_text() {
        let pipeline = make_pipeline(ReplyMode::MatchInput);
        let result = pipeline
            .synthesize_response("test response", false)
            .await
            .unwrap();
        assert!(matches!(result, VoiceOutput::TextOnly(_)));
    }

    #[tokio::test]
    async fn synthesize_response_voice_only_always_voices() {
        let pipeline = make_pipeline(ReplyMode::VoiceOnly);
        let result = pipeline.synthesize_response("test", false).await.unwrap();
        assert!(matches!(result, VoiceOutput::WithAudio { .. }));
    }

    #[tokio::test]
    async fn health_check_reports_both() {
        let pipeline = make_pipeline(ReplyMode::MatchInput);
        let (stt, tts) = pipeline.health_check().await;
        assert!(stt);
        assert!(tts);
    }
}
