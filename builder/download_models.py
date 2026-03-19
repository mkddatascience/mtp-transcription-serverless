import os

# Set HuggingFace cache directory (same as: export HF_HOME=...)
os.environ["HF_HOME"] = "./downloaded_models"

from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or "hf...."

MODELS = [
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ForcedAligner-0.6B",
    "pyannote/speaker-diarization-3.1",
]

for model in MODELS:
    print(f"Downloading {model} ...")

    snapshot_download(
        repo_id=model,
        token=HF_TOKEN,
        resume_download=True
    )

print("All models downloaded...")