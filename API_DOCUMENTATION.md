# Qwen3-ASR RunPod API Documentation

## Endpoints

### 1. Submit Job
**URL:** `https://api.runpod.ai/v2/7pekfthk43ua24/run`  
**Method:** `POST`  
**Authentication:** Bearer Token (Required)

### 2. Check Status
**URL:** `https://api.runpod.ai/v2/7pekfthk43ua24/status/{id}`  
**Method:** `GET`  
**Authentication:** Bearer Token (Required)

---

## Authentication

Include your RunPod API key in the request headers:

```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

---

## Request Payload

### Structure

```json
{
  "input": {
    "audio": "string (required)",
    "language": "string (optional)",
    "return_time_stamps": true,
    "enable_diarization": true,
    "num_speakers": null,
    "min_speakers": null,
    "max_speakers": null,
    "max_new_tokens": 256
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio` | string | **Yes** | - | URL to the audio file (WAV, MP3, FLAC, etc.) |
| `language` | string | No | `null` | Language code (e.g., "en", "es", "fr"). Auto-detects if null |
| `return_time_stamps` | boolean | No | `true` | Include word-level timestamps in response |
| `enable_diarization` | boolean | No | `true` | Enable speaker diarization |
| `num_speakers` | integer | No | `null` | Exact number of speakers (overrides min/max) |
| `min_speakers` | integer | No | `null` | Minimum number of speakers to detect |
| `max_speakers` | integer | No | `null` | Maximum number of speakers to detect |
| `max_new_tokens` | integer | No | `256` | Maximum tokens for generation |

### Example Payloads

**Basic transcription:**
```json
{
  "input": {
    "audio": "https://example.com/audio.wav",
    "enable_diarization": false
  }
}
```

**With speaker diarization:**
```json
{
  "input": {
    "audio": "https://files.pyannote.ai/marklex1min.wav",
    "enable_diarization": true
  }
}
```

**With specific number of speakers:**
```json
{
  "input": {
    "audio": "https://example.com/meeting.wav",
    "enable_diarization": true,
    "num_speakers": 3
  }
}
```

**With speaker range:**
```json
{
  "input": {
    "audio": "https://example.com/podcast.wav",
    "enable_diarization": true,
    "min_speakers": 2,
    "max_speakers": 4,
    "language": "en"
  }
}
```

---

## Responses

### Job Submission Response

When you submit a job to `/run`, you receive:

```json
{
  "id": "dbdc359b-2a96-43df-9ec0-b0d708de1704-e1",
  "status": "IN_QUEUE"
}
```

### Job Status Values

| Status | Description |
|--------|-------------|
| `IN_QUEUE` | Job is queued and waiting to be processed |
| `IN_PROGRESS` | Job is currently being processed |
| `COMPLETED` | Job completed successfully |
| `FAILED` | Job failed with an error |

### Status Response Examples

**IN_QUEUE:**
```json
{
  "id": "dbdc359b-2a96-43df-9ec0-b0d708de1704-e1",
  "status": "IN_QUEUE"
}
```

**IN_PROGRESS:**
```json
{
  "id": "dbdc359b-2a96-43df-9ec0-b0d708de1704-e1",
  "status": "IN_PROGRESS"
}
```

**COMPLETED:**
```json
{
  "id": "dbdc359b-2a96-43df-9ec0-b0d708de1704-e1",
  "status": "COMPLETED",
  "output": {
    "detected_language": "en",
    "transcription": "Full transcription text here...",
    "time_stamps": [
      {
        "text": "Hello",
        "start": 0.0,
        "end": 0.5
      },
      {
        "text": "world",
        "start": 0.5,
        "end": 1.0
      }
    ],
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello world, this is a test.",
        "speaker": "SPEAKER_00"
      },
      {
        "start": 3.0,
        "end": 5.2,
        "text": "Yes, I can hear you clearly.",
        "speaker": "SPEAKER_01"
      }
    ]
  },
  "executionTime": 5203,
  "workerId": "worker-id"
}
```

**FAILED:**
```json
{
  "id": "dbdc359b-2a96-43df-9ec0-b0d708de1704-e1",
  "status": "FAILED",
  "error": "Error message here",
  "executionTime": 1234,
  "workerId": "worker-id"
}
```

---

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `detected_language` | string | Auto-detected language code |
| `transcription` | string | Complete transcription text |
| `time_stamps` | array | Word-level timestamps (if `return_time_stamps` is true) |
| `time_stamps[].text` | string | Individual word |
| `time_stamps[].start` | float | Word start time in seconds |
| `time_stamps[].end` | float | Word end time in seconds |
| `segments` | array | Phrase-level segments with speaker labels |
| `segments[].start` | float | Segment start time in seconds |
| `segments[].end` | float | Segment end time in seconds |
| `segments[].text` | string | Transcribed text for the segment |
| `segments[].speaker` | string | Speaker label (e.g., "SPEAKER_00", "SPEAKER_01") |

---

## Workflow

1. **Submit job** to `/run` endpoint with audio URL and parameters
2. **Receive job ID** and initial status (`IN_QUEUE`)
3. **Poll status** at `/status/{id}` endpoint every 2-5 seconds
4. **Check status** until it becomes `COMPLETED` or `FAILED`
5. **Retrieve output** from the response when status is `COMPLETED`

**Status progression:**
```
IN_QUEUE → IN_PROGRESS → COMPLETED/FAILED
```

---

## Notes

- **Audio Requirements:** Audio file must be publicly accessible via URL
- **Supported Formats:** WAV, MP3, FLAC, and other common audio formats
- **Processing Time:** Typically 10-30% of audio duration
- **Polling Interval:** Check status every 2-5 seconds
- **Speaker Diarization:** Most accurate with 2-10 speakers
- **Authentication:** All requests require Bearer token in Authorization header

---

## Troubleshooting

### 401 Unauthorized
- Verify your Bearer token is correct
- Ensure Authorization header is properly formatted

### 400 Bad Request
- Check that the audio URL is accessible
- Verify all required fields are present
- Ensure JSON payload is properly formatted

### Timeout Errors
- Very long audio files may timeout
- Consider splitting audio into smaller chunks

### Speaker Diarization Issues
- Ensure `enable_diarization` is set to `true`
- Verify HF_TOKEN is configured on the server
- Audio quality affects diarization accuracy
