pub mod assemblyai;
pub mod inworld;
pub mod pipeline;
pub mod traits;

pub use pipeline::DefaultVoicePipeline;
// Boundary types re-exported from channels::traits (canonical home).
#[allow(unused_imports)]
pub use crate::channels::traits::{AudioFormat, AudioOutput, VoiceAttachment, VoiceOutput};
pub use traits::{ReplyMode, SttProvider, TtsProvider, VoicePipeline, VoiceProfile};
// Streaming types — re-exported for use by future streaming pipeline implementations.
#[allow(unused_imports)]
pub use traits::{AudioChunk, StreamingSttProvider, StreamingTtsProvider, TranscriptChunk};

use crate::config::schema::VoiceConfig;

/// Build a `VoicePipeline` from configuration, resolving API keys from
/// inline config or environment variables.
pub fn create_voice_pipeline(
    config: &VoiceConfig,
) -> anyhow::Result<Option<Box<dyn VoicePipeline>>> {
    if !config.enabled {
        return Ok(None);
    }

    // --- STT provider ---
    let stt: Box<dyn SttProvider> = match config.stt.provider.as_str() {
        "assemblyai" => {
            let api_key =
                resolve_api_key(&config.stt.api_key, &config.stt.api_key_env, "ASSEMBLYAI_API_KEY")?;
            Box::new(assemblyai::AssemblyAiStt::new(api_key))
        }
        other => anyhow::bail!("Unsupported STT provider: {other}"),
    };

    // --- TTS provider ---
    let tts: Box<dyn TtsProvider> = match config.tts.provider.as_str() {
        "inworld" => {
            let api_key =
                resolve_api_key(&config.tts.api_key, &config.tts.api_key_env, "INWORLD_API_KEY")?;
            Box::new(inworld::InworldTts::new(
                api_key,
                config.tts.voice_id.clone(),
                config.tts.model.clone(),
            ))
        }
        other => anyhow::bail!("Unsupported TTS provider: {other}"),
    };

    let output_format = config
        .tts
        .output_format
        .as_deref()
        .map(parse_audio_format)
        .transpose()?
        .unwrap_or(AudioFormat::OggOpus);

    let voice_profile = VoiceProfile {
        voice_id: config.tts.voice_id.clone(),
        output_format,
        speed: None,
    };

    let reply_mode = config
        .reply_mode
        .as_deref()
        .map(parse_reply_mode)
        .transpose()?
        .unwrap_or(ReplyMode::MatchInput);

    Ok(Some(Box::new(DefaultVoicePipeline::new(
        stt,
        tts,
        voice_profile,
        reply_mode,
    ))))
}

/// Resolve an API key: inline config value takes priority, then env var.
fn resolve_api_key(
    inline_key: &Option<String>,
    env_name: &Option<String>,
    default_env: &str,
) -> anyhow::Result<String> {
    // 1. Inline api_key in config.toml
    if let Some(key) = inline_key {
        if !key.is_empty() {
            return Ok(key.clone());
        }
    }
    // 2. Environment variable
    let var_name = env_name.as_deref().unwrap_or(default_env);
    std::env::var(var_name).map_err(|_| {
        anyhow::anyhow!(
            "Voice provider API key not found. Set the {var_name} environment variable \
             or provide api_key in the [voice] config section."
        )
    })
}

fn parse_audio_format(s: &str) -> anyhow::Result<AudioFormat> {
    match s.to_lowercase().as_str() {
        "ogg_opus" | "ogg" | "opus" => Ok(AudioFormat::OggOpus),
        "mp3" => Ok(AudioFormat::Mp3),
        "wav" => Ok(AudioFormat::Wav),
        "webm" => Ok(AudioFormat::Webm),
        other => anyhow::bail!("Unknown audio format: {other}"),
    }
}

fn parse_reply_mode(s: &str) -> anyhow::Result<ReplyMode> {
    match s.to_lowercase().as_str() {
        "voice_only" => Ok(ReplyMode::VoiceOnly),
        "text_and_voice" => Ok(ReplyMode::TextAndVoice),
        "match_input" => Ok(ReplyMode::MatchInput),
        other => anyhow::bail!("Unknown voice reply mode: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_audio_format_variants() {
        assert_eq!(
            parse_audio_format("ogg_opus").unwrap(),
            AudioFormat::OggOpus
        );
        assert_eq!(parse_audio_format("mp3").unwrap(), AudioFormat::Mp3);
        assert_eq!(parse_audio_format("wav").unwrap(), AudioFormat::Wav);
        assert!(parse_audio_format("unknown").is_err());
    }

    #[test]
    fn parse_reply_mode_variants() {
        assert_eq!(
            parse_reply_mode("match_input").unwrap(),
            ReplyMode::MatchInput
        );
        assert_eq!(
            parse_reply_mode("voice_only").unwrap(),
            ReplyMode::VoiceOnly
        );
        assert!(parse_reply_mode("bad").is_err());
    }

    #[test]
    fn create_pipeline_disabled() {
        let config = VoiceConfig::default();
        let result = create_voice_pipeline(&config).unwrap();
        assert!(result.is_none());
    }
}
