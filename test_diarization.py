"""
Standalone test script for the pyannote speaker diarization pipeline.
Runs diarization only — no ASR/Qwen3 involved.

Usage:
    python test_diarization.py                          # generates a synthetic test audio
    python test_diarization.py --audio path/to/file.wav # use your own file
    python test_diarization.py --audio file.wav --num_speakers 2

Requires:
    pip install pyannote.audio torch numpy scipy soundfile

Environment variables:
    HF_TOKEN - HuggingFace token with access to pyannote/speaker-diarization-3.1
"""

import os
import argparse
import tempfile

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")


# ── Synthetic audio generator ──────────────────────────────────────────────

def generate_synthetic_audio(output_path: str, sample_rate: int = 16000):
    """
    Generate a simple synthetic audio file with two 'speakers' (different
    sine wave tones separated by short silences) for pipeline smoke-testing.

    Speaker 1: 200 Hz tone for 3 seconds
    Silence:   0.5 seconds
    Speaker 2: 400 Hz tone for 3 seconds
    Silence:   0.5 seconds
    Speaker 1: 200 Hz tone for 2 seconds
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required: pip install soundfile")

    def tone(freq, duration, sr=sample_rate, amplitude=0.3):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    audio = np.concatenate([
        tone(200, 3.0),   # Speaker 1
        silence,
        tone(400, 3.0),   # Speaker 2
        silence,
        tone(200, 2.0),   # Speaker 1 again
    ])

    sf.write(output_path, audio, sample_rate)
    print(f"Synthetic audio written to: {output_path}")
    print(f"  Duration    : {len(audio) / sample_rate:.2f}s")
    print(f"  Sample rate : {sample_rate} Hz")
    return output_path


# ── Diarization runner ─────────────────────────────────────────────────────

MODEL_DIR = os.getenv(
    "DIARIZATION_MODEL_DIR",
    "./downloaded_models/pyannote_speaker-diarization-3.1"
)
def run_diarization(
    audio_path: str,
    hf_token: str,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
):
    from pyannote.audio import Pipeline

    # Use global MODEL_DIR
    print(f"Loading diarization model from: {MODEL_DIR}")

    pipeline = Pipeline.from_pretrained(
     "pyannote/speaker-diarization-3.1",  # just the folder path
     token=hf_token
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Pipeline moved to GPU.")
    else:
        print("Running on CPU (no CUDA detected).")

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    print(f"Running diarization on: {audio_path}")
    if diarize_kwargs:
        print(f"  Speaker hints: {diarize_kwargs}")

    diarization = pipeline(audio_path, **diarize_kwargs)
    print("TYPE:", type(diarization))
    print("ATTRS:", [a for a in dir(diarization) if not a.startswith('_')])
    turns = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    return turns


# ── Pretty printer ─────────────────────────────────────────────────────────

def print_results(turns):
    if not turns:
        print("\nNo diarization turns returned.")
        return

    speakers = sorted(set(t["speaker"] for t in turns))
    print(f"\n{'='*50}")
    print(f"Diarization Results")
    print(f"{'='*50}")
    print(f"Unique speakers detected : {len(speakers)} → {', '.join(speakers)}")
    print(f"Total segments           : {len(turns)}\n")

    for i, turn in enumerate(turns, 1):
        duration = round(turn["end"] - turn["start"], 3)
        print(f"  [{i:>3}] {turn['speaker']}  "
              f"{turn['start']:>7.3f}s → {turn['end']:>7.3f}s  "
              f"(dur: {duration:.3f}s)")

    print(f"\nPer-speaker total speaking time:")
    from collections import defaultdict
    totals = defaultdict(float)
    for t in turns:
        totals[t["speaker"]] += t["end"] - t["start"]
    for spk in sorted(totals):
        print(f"  {spk}: {totals[spk]:.2f}s")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone pyannote diarization test"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="0a8d813b-b158-4783-80fe-40604d3e66db.wav",
        help="Path to audio file. If omitted, a synthetic test file is generated.",
    )
    parser.add_argument(
        "--num_speakers", type=int, default=None,
        help="Fix exact number of speakers.",
    )
    parser.add_argument(
        "--min_speakers", type=int, default=None,
        help="Minimum number of speakers hint.",
    )
    parser.add_argument(
        "--max_speakers", type=int, default=None,
        help="Maximum number of speakers hint.",
    )
    parser.add_argument(
        "--hf_token", type=str, default=HF_TOKEN,
        help="HuggingFace token (overrides HF_TOKEN env var).",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError(
            "HF_TOKEN is required. Set it as an env var or pass --hf_token."
        )

    tmp_file = None
    if args.audio is None:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.close()
        audio_path = generate_synthetic_audio(tmp_file.name)
    else:
        audio_path = args.audio
        print(f"Using provided audio file: {audio_path}")

    try:
        turns = run_diarization(
            audio_path=audio_path,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        print_results(turns)
    finally:
        if tmp_file is not None and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
            print(f"\nTemp file cleaned up.")


if __name__ == "__main__":
    main()