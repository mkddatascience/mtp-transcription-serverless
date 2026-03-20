"""
Predictor class for Qwen3-ASR model with optional speaker diarization
powered by pyannote.audio.

Environment variables:
    MODEL_NAME       - Qwen3 ASR model (default: Qwen/Qwen3-ASR-1.7B)
    FORCED_ALIGNER   - Forced aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B)
    HF_TOKEN         - HuggingFace token for pyannote diarization
"""

import os
import time
import torch
from dotenv import load_dotenv

os.environ["HF_HOME"] = "/downloaded_models"
load_dotenv()

MODEL_NAME     = os.getenv("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
FORCED_ALIGNER = os.getenv("FORCED_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")
HF_TOKEN       = os.getenv("HF_TOKEN", "HF_TOKEN_HERE")

SEGMENT_GAP_THRESHOLD = 1.0
MIN_DIARIZATION_SECONDS = 10


class Predictor:
    """Qwen3-ASR with optional pyannote speaker diarization."""

    def setup(self, enable_diarization=True):
        """Load the Qwen3-ASR model and optionally the pyannote diarization
        pipeline. Both are loaded once and reused across requests.
        """
        setup_start = time.time()
        from qwen_asr import Qwen3ASRModel

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.diarization_pipeline = None

        # ── Load Qwen3-ASR ─────────────────────────────────────────────────
        print(f"Loading Qwen3-ASR model: {MODEL_NAME} on {self.device}")
        model_load_start = time.time()

        self.model = Qwen3ASRModel.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map=self.device,
            max_inference_batch_size=32,
            max_new_tokens=256,
            forced_aligner=FORCED_ALIGNER,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=self.device,
            ),
        )
        model_load_time = time.time() - model_load_start
        print(f"Qwen3-ASR model loaded successfully in {model_load_time:.2f}s")

        # ── Load pyannote diarization ───────────────────────────────────────
        if enable_diarization and HF_TOKEN:
            try:
                diarization_load_start = time.time()
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN,
                )
                if torch.cuda.is_available():
                    self.diarization_pipeline.to(torch.device("cuda"))
                diarization_load_time = time.time() - diarization_load_start
                print(f"Speaker diarization pipeline loaded successfully in {diarization_load_time:.2f}s")
            except Exception as e:
                print(f"Could not load diarization pipeline: {e}")
        elif not enable_diarization:
            print("Speaker diarization disabled by configuration.")
        else:
            print("No HF_TOKEN found. Speaker diarization disabled.")

        total_setup_time = time.time() - setup_start
        print(f"Total setup time: {total_setup_time:.2f}s")

    def predict(
        self,
        audio,
        language=None,
        return_time_stamps=True,
        enable_diarization=True,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        max_new_tokens=256,
    ):
        """
        Transcribe one or more audio files with optional speaker diarization.

        Returns:
            List of dicts with: detected_language, transcription,
                                time_stamps, segments, diarization_skipped,
                                diarization_skip_reason
        """
        predict_start = time.time()
        timing_breakdown = {}

        prep_start = time.time()
        if isinstance(audio, str):
            audio = [audio]

        if language is None:
            language_list = [None] * len(audio)
        elif isinstance(language, str):
            language_list = [language] * len(audio)
        else:
            language_list = language
        timing_breakdown['preparation'] = time.time() - prep_start

        print(f"Transcribing {len(audio)} file(s) with Qwen3-ASR...")
        transcribe_start = time.time()

        results = self.model.transcribe(
            audio=audio,
            language=language_list,
            return_time_stamps=True,
        )
        timing_breakdown['transcription'] = time.time() - transcribe_start
        print(f"Transcription completed in {timing_breakdown['transcription']:.2f}s")

        outputs = []
        print(f"Results: {results}")

        for idx, (audio_path, r) in enumerate(zip(audio, results)):
            segment_start = time.time()

            grouping_start = time.time()
            segments = _group_words_into_segments(
                words=r.time_stamps.items if r.time_stamps else [],
                gap_threshold=SEGMENT_GAP_THRESHOLD,
            )
            timing_breakdown[f'file_{idx}_word_grouping'] = time.time() - grouping_start

            diarization_skipped = False
            diarization_skip_reason = None

            if enable_diarization and self.diarization_pipeline is not None:
                # Validate audio before attempting diarization
                is_valid, reason = _validate_audio_for_diarization(audio_path)

                if not is_valid:
                    diarization_skipped = True
                    diarization_skip_reason = reason
                    print(f"Diarization skipped for file {idx}: {reason}")
                    # Mark all segments as UNKNOWN speaker
                    for segment in segments:
                        segment["speaker"] = "UNKNOWN"
                else:
                    diarize_start = time.time()
                    segments, diarization_skipped, diarization_skip_reason = self._apply_diarization(
                        audio_path=audio_path,
                        segments=segments,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    timing_breakdown[f'file_{idx}_diarization'] = time.time() - diarize_start
                    print(f"Diarization for file {idx} completed in {timing_breakdown[f'file_{idx}_diarization']:.2f}s")
            elif enable_diarization and self.diarization_pipeline is None:
                diarization_skipped = True
                diarization_skip_reason = "Diarization pipeline not loaded (missing HF_TOKEN or load failed)"
                for segment in segments:
                    segment["speaker"] = "UNKNOWN"

            serialize_start = time.time()
            outputs.append({
                "detected_language":       r.language,
                "transcription":           r.text,
                "time_stamps":             _serialize_timestamps(r.time_stamps) if return_time_stamps else None,
                "segments":                segments,
                "diarization_skipped":     diarization_skipped,
                "diarization_skip_reason": diarization_skip_reason,
            })
            timing_breakdown[f'file_{idx}_serialization'] = time.time() - serialize_start
            timing_breakdown[f'file_{idx}_total'] = time.time() - segment_start

        total_time = time.time() - predict_start
        timing_breakdown['total_predict_time'] = total_time

        print("\n" + "="*60)
        print("TIMING BREAKDOWN:")
        print("="*60)
        for key, value in timing_breakdown.items():
            print(f"  {key:.<50} {value:>6.2f}s")
        print("="*60 + "\n")

        return outputs

    def _apply_diarization(
        self,
        audio_path,
        segments,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        """
        Run pyannote diarization and assign speaker labels to each segment
        using maximum temporal overlap.

        Returns:
            (segments, diarization_skipped, diarization_skip_reason)
        """
        print(f"Running speaker diarization on: {audio_path}")
        diarize_timing = {}

        kwargs_start = time.time()
        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
        diarize_timing['kwargs_prep'] = time.time() - kwargs_start

        try:
            pipeline_start = time.time()
            diarization = self.diarization_pipeline(audio_path, **diarize_kwargs)
            diarize_timing['pipeline_execution'] = time.time() - pipeline_start
            print(f"  Pipeline execution: {diarize_timing['pipeline_execution']:.2f}s")

            extract_start = time.time()
            diarization_turns = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker
                in diarization.speaker_diarization.itertracks(yield_label=True)
            ]
            diarize_timing['extract_turns'] = time.time() - extract_start

            assign_start = time.time()
            for segment in segments:
                segment["speaker"] = _assign_speaker(
                    segment["start"], segment["end"], diarization_turns
                )
            diarize_timing['assign_speakers'] = time.time() - assign_start

            print(f"  Diarization timing breakdown:")
            for key, value in diarize_timing.items():
                print(f"    {key}: {value:.3f}s")

            return segments, False, None

        except Exception as e:
            reason = f"Diarization failed during pipeline execution: {str(e)}"
            print(f"  ERROR: {reason}")
            for segment in segments:
                segment["speaker"] = "UNKNOWN"
            return segments, True, reason


# ── Audio validation ───────────────────────────────────────────────────────

def _validate_audio_for_diarization(audio_path: str):
    """
    Validate audio file is suitable for pyannote diarization.

    Returns:
        (is_valid: bool, reason: str or None)
        reason is None if valid, otherwise a human-readable explanation.
    """
    import soundfile as sf

    # Check file exists
    if not os.path.exists(audio_path):
        return False, f"Audio file does not exist: {audio_path}"

    # Check file is not empty
    if os.path.getsize(audio_path) == 0:
        return False, "Audio file is empty (0 bytes)"

    # Check file is readable and get duration
    try:
        info = sf.info(audio_path)
    except Exception as e:
        return False, f"Audio file is unreadable or corrupted: {str(e)}"

    # Check sample rate
    if info.samplerate != 48000:
        return False, (
            f"Audio sample rate is {info.samplerate}Hz, expected 48000Hz. "
            f"Normalization may have failed."
        )

    # Check duration
    duration_seconds = info.frames / info.samplerate
    if duration_seconds < MIN_DIARIZATION_SECONDS:
        return False, (
            f"Audio duration is {duration_seconds:.2f}s, minimum required for "
            f"diarization is {MIN_DIARIZATION_SECONDS}s."
        )

    # Check sample count matches what pyannote expects for first chunk
    expected_samples = info.samplerate * MIN_DIARIZATION_SECONDS  # 480000
    if info.frames < expected_samples:
        return False, (
            f"Audio has {info.frames} samples, pyannote requires at least "
            f"{expected_samples} samples ({MIN_DIARIZATION_SECONDS}s at {info.samplerate}Hz)."
        )

    return True, None


# ── Word grouping ──────────────────────────────────────────────────────────

def _group_words_into_segments(words, gap_threshold=1.0):
    """Group word-level ForcedAlignItems into phrase-level segments."""
    if not words:
        return []

    segments = []
    current_words = [words[0]]

    for word in words[1:]:
        gap = word.start_time - current_words[-1].end_time
        if gap > gap_threshold:
            segments.append(_words_to_segment(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        segments.append(_words_to_segment(current_words))

    return segments


def _words_to_segment(words):
    """Convert a list of ForcedAlignItems into a segment dict."""
    return {
        "start":   words[0].start_time,
        "end":     words[-1].end_time,
        "text":    " ".join(w.text for w in words),
        "speaker": "SPEAKER_00",
    }


# ── Speaker assignment ─────────────────────────────────────────────────────

def _assign_speaker(seg_start, seg_end, diarization_turns):
    """Return the speaker with the greatest temporal overlap with the segment."""
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn_start, turn_end, speaker in diarization_turns:
        overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


# ── Output formatters ──────────────────────────────────────────────────────

def format_timestamp(seconds, always_include_hours=False, decimal_marker="."):
    """Convert seconds to SRT/VTT timestamp string."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    if always_include_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"
    return f"{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def write_plain_with_speakers(segments):
    return "".join(
        f"[{seg.get('speaker', 'UNKNOWN')}]: {seg['text'].strip()}\n"
        for seg in segments
    )


def write_vtt(segments):
    result = ""
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        result += f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n"
        result += f"<v {speaker}>{seg['text'].strip().replace('-->', '->')}\n\n"
    return result


def write_srt(segments):
    result = ""
    for i, seg in enumerate(segments, start=1):
        speaker = seg.get("speaker", "UNKNOWN")
        result += f"{i}\n"
        result += (
            f"{format_timestamp(seg['start'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(seg['end'], always_include_hours=True, decimal_marker=',')}\n"
        )
        result += f"[{speaker}] {seg['text'].strip().replace('-->', '->')}\n\n"
    return result


def _serialize_timestamps(time_stamps):
    """Convert ForcedAlignResult to JSON serializable format."""
    if time_stamps is None:
        return None
    try:
        return [
            {
                "text":  item.text,
                "start": item.start_time,
                "end":   item.end_time,
            }
            for item in time_stamps.items
        ]
    except Exception:
        return None