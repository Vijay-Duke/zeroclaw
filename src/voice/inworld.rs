use crate::channels::traits::{AudioFormat, AudioOutput};
use super::traits::{TtsProvider, VoiceProfile};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::Deserialize;

const INWORLD_TTS_URL: &str = "https://api.inworld.ai/tts/v1/voice";

/// Maximum text length per Inworld API request (characters).
const MAX_TEXT_LENGTH: usize = 2000;

/// Default TTS model.
const DEFAULT_MODEL: &str = "inworld-tts-1.5-max";

/// Default voice when none is configured.
const DEFAULT_VOICE: &str = "Dennis";

#[derive(Debug, Deserialize)]
struct TtsResponse {
    #[serde(rename = "audioContent")]
    audio_content: String, // base64-encoded audio
}

/// Map our `AudioFormat` to Inworld's encoding enum values.
fn audio_format_to_inworld_encoding(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::OggOpus => "OGG_OPUS",
        AudioFormat::Mp3 => "MP3",
        AudioFormat::Wav => "LINEAR16",
        AudioFormat::Webm => "MP3", // Inworld doesn't support webm; fallback to MP3
    }
}

/// Inworld AI text-to-speech provider.
///
/// Uses Inworld's TTS API to synthesize speech with character voices.
/// ZeroClaw provides the text (it's the brain); Inworld provides the voice
/// personality and vocal delivery.
pub struct InworldTts {
    client: reqwest::Client,
    /// Base64-encoded API key credentials for Basic auth.
    api_credential: String,
    /// Default voice ID (e.g. "Dennis", "Celeste").
    default_voice: String,
    /// TTS model ID.
    model: String,
}

impl InworldTts {
    /// Create a new Inworld TTS provider.
    ///
    /// `api_key` can be either:
    /// - A pre-encoded base64 credential string (from Inworld Studio)
    /// - A raw `workspace_id:api_secret` pair (will be base64-encoded automatically)
    pub fn new(api_key: String, default_voice: Option<String>, model: Option<String>) -> Self {
        // Detect whether the key is already base64-encoded.
        // If it decodes cleanly and contains ':', it's pre-encoded credentials.
        let api_credential = if BASE64.decode(&api_key).is_ok() && !api_key.contains(':') {
            // Already base64-encoded — use as-is for Basic auth
            api_key
        } else {
            // Raw key — base64-encode for Basic auth
            BASE64.encode(api_key.as_bytes())
        };
        Self {
            client: reqwest::Client::new(),
            api_credential,
            default_voice: default_voice.unwrap_or_else(|| DEFAULT_VOICE.into()),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.into()),
        }
    }
}

#[async_trait]
impl TtsProvider for InworldTts {
    fn name(&self) -> &str {
        "inworld"
    }

    async fn synthesize(&self, text: &str, profile: &VoiceProfile) -> anyhow::Result<AudioOutput> {
        if text.is_empty() {
            anyhow::bail!("Cannot synthesize empty text");
        }

        // Truncate to Inworld's limit to avoid API errors.
        let text = if text.len() > MAX_TEXT_LENGTH {
            tracing::warn!(
                "TTS text truncated from {} to {MAX_TEXT_LENGTH} characters",
                text.len()
            );
            &text[..MAX_TEXT_LENGTH]
        } else {
            text
        };

        let voice_id = profile.voice_id.as_deref().unwrap_or(&self.default_voice);

        let encoding = audio_format_to_inworld_encoding(profile.output_format);

        let body = serde_json::json!({
            "text": text,
            "voiceId": voice_id,
            "modelId": &self.model,
            "audioConfig": {
                "audioEncoding": encoding,
            },
        });

        let resp = self
            .client
            .post(INWORLD_TTS_URL)
            .header("authorization", format!("Basic {}", self.api_credential))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let tts_resp: TtsResponse = resp.json().await?;

        let audio_bytes = BASE64
            .decode(&tts_resp.audio_content)
            .map_err(|e| anyhow::anyhow!("Failed to decode Inworld audio response: {e}"))?;

        Ok(AudioOutput {
            data: audio_bytes,
            format: profile.output_format,
            duration_ms: None, // Inworld doesn't return duration in the basic response
        })
    }

    async fn health_check(&self) -> bool {
        // Synthesize a minimal phrase to check connectivity.
        // This is lightweight — Inworld bills by character count.
        let profile = VoiceProfile {
            voice_id: Some(self.default_voice.clone()),
            output_format: AudioFormat::Mp3,
            speed: None,
        };
        self.synthesize("ok", &profile).await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inworld_tts_name() {
        let tts = InworldTts::new("test-key".into(), None, None);
        assert_eq!(tts.name(), "inworld");
    }

    #[test]
    fn audio_format_mapping() {
        assert_eq!(
            audio_format_to_inworld_encoding(AudioFormat::OggOpus),
            "OGG_OPUS"
        );
        assert_eq!(audio_format_to_inworld_encoding(AudioFormat::Mp3), "MP3");
        assert_eq!(
            audio_format_to_inworld_encoding(AudioFormat::Wav),
            "LINEAR16"
        );
    }

    #[test]
    fn default_voice_and_model() {
        let tts = InworldTts::new("key".into(), None, None);
        assert_eq!(tts.default_voice, "Dennis");
        assert_eq!(tts.model, "inworld-tts-1.5-max");
    }

    #[test]
    fn custom_voice_and_model() {
        let tts = InworldTts::new("key".into(), Some("Celeste".into()), Some("tts-1".into()));
        assert_eq!(tts.default_voice, "Celeste");
        assert_eq!(tts.model, "tts-1");
    }
}
