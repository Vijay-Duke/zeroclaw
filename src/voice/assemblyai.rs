use crate::channels::traits::AudioFormat;
use super::traits::{SttProvider, Transcript};
use async_trait::async_trait;
use serde::Deserialize;

const ASSEMBLYAI_UPLOAD_URL: &str = "https://api.assemblyai.com/v2/upload";
const ASSEMBLYAI_TRANSCRIPT_URL: &str = "https://api.assemblyai.com/v2/transcript";

/// Maximum time to wait for a transcription job to complete.
const TRANSCRIPTION_TIMEOUT_SECS: u64 = 120;
/// Polling interval for transcription status.
const POLL_INTERVAL_MS: u64 = 1500;

#[derive(Debug, Deserialize)]
struct UploadResponse {
    upload_url: String,
}

#[derive(Debug, Deserialize)]
struct TranscriptResponse {
    id: String,
    status: String,
    text: Option<String>,
    language_code: Option<String>,
    confidence: Option<f64>,
    error: Option<String>,
}

/// AssemblyAI speech-to-text provider.
///
/// Uses AssemblyAI's upload + transcription API:
/// 1. Upload audio bytes → get a temporary URL
/// 2. Submit transcription request with that URL
/// 3. Poll until transcription completes
pub struct AssemblyAiStt {
    client: reqwest::Client,
    api_key: String,
}

impl AssemblyAiStt {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    fn auth_header(&self) -> (&str, &str) {
        ("authorization", self.api_key.as_str())
    }

    /// Upload audio bytes to AssemblyAI's temporary storage.
    async fn upload_audio(&self, audio: &[u8]) -> anyhow::Result<String> {
        let resp: UploadResponse = self
            .client
            .post(ASSEMBLYAI_UPLOAD_URL)
            .header(self.auth_header().0, self.auth_header().1)
            .header("content-type", "application/octet-stream")
            .body(audio.to_vec())
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.upload_url)
    }

    /// Submit a transcription job and return the transcript ID.
    async fn submit_transcription(&self, audio_url: &str) -> anyhow::Result<String> {
        let body = serde_json::json!({
            "audio_url": audio_url,
            "language_detection": true,
        });

        let resp: TranscriptResponse = self
            .client
            .post(ASSEMBLYAI_TRANSCRIPT_URL)
            .header(self.auth_header().0, self.auth_header().1)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        Ok(resp.id)
    }

    /// Poll a transcription job until it completes or fails.
    async fn poll_transcription(&self, transcript_id: &str) -> anyhow::Result<TranscriptResponse> {
        let url = format!("{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}");
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(TRANSCRIPTION_TIMEOUT_SECS);

        loop {
            if tokio::time::Instant::now() > deadline {
                anyhow::bail!(
                    "AssemblyAI transcription timed out after {TRANSCRIPTION_TIMEOUT_SECS}s"
                );
            }

            let resp: TranscriptResponse = self
                .client
                .get(&url)
                .header(self.auth_header().0, self.auth_header().1)
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            match resp.status.as_str() {
                "completed" => return Ok(resp),
                "error" => {
                    let detail = resp.error.unwrap_or_else(|| "unknown error".into());
                    anyhow::bail!("AssemblyAI transcription failed: {detail}");
                }
                _ => {
                    // "queued" or "processing" — wait and retry
                    tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
                }
            }
        }
    }
}

#[async_trait]
impl SttProvider for AssemblyAiStt {
    fn name(&self) -> &str {
        "assemblyai"
    }

    async fn transcribe(&self, audio: &[u8], _format: AudioFormat) -> anyhow::Result<Transcript> {
        // Step 1: Upload audio
        let upload_url = self.upload_audio(audio).await?;

        // Step 2: Submit transcription
        let transcript_id = self.submit_transcription(&upload_url).await?;

        // Step 3: Poll for result
        let result = self.poll_transcription(&transcript_id).await?;

        Ok(Transcript {
            text: result.text.unwrap_or_default(),
            language: result.language_code,
            confidence: result.confidence.map(|c| c as f32),
        })
    }

    async fn health_check(&self) -> bool {
        // Quick check: attempt to access the API with a minimal request.
        // AssemblyAI doesn't have a dedicated health endpoint, so we
        // check if authentication works by hitting the transcript list.
        self.client
            .get(ASSEMBLYAI_TRANSCRIPT_URL)
            .header(self.auth_header().0, self.auth_header().1)
            .query(&[("limit", "1")])
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemblyai_stt_name() {
        let stt = AssemblyAiStt::new("test-key".into());
        assert_eq!(stt.name(), "assemblyai");
    }
}
