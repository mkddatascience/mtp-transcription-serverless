"""
Local download utility for audio URLs.
More robust than runpod's rp_download for URLs like files.pyannote.ai:
- Longer timeouts (5s is too short for ~14MB files)
- Content-Type fallback for extension when URL has none
- All audio is normalized to 48kHz mono WAV via ffmpeg (pyannote expects 48kHz chunks)
"""

import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Dict
from urllib.parse import urlparse

import backoff
import requests
import soundfile as sf
from requests import RequestException

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}

CONTENT_TYPE_TO_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
}

TIMEOUT = (30, 180)

WAV_SAMPLE_RATE = 48000
WAV_CHANNELS = 1
MIN_DIARIZATION_SECONDS = 10
TARGET_SAMPLES = WAV_SAMPLE_RATE * MIN_DIARIZATION_SECONDS  # 480000

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".webm", ".flac", ".wav", ".bin"}


def normalize_to_wav(audio_path: str) -> str:
    """
    Convert any audio to 48kHz mono 16-bit WAV via ffmpeg.
    Pads to exactly 480000 samples (10s at 48kHz) minimum.
    Uses pad_len instead of whole_dur to avoid sample count drift.
    Removes -fflags +genpts which caused timestamp jitter.
    Returns path to the .wav file. Removes original if different.
    """
    base, ext = os.path.splitext(audio_path)
    wav_path = base + ".wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ar", str(WAV_SAMPLE_RATE),
        "-ac", str(WAV_CHANNELS),
        "-af", f"apad=pad_len={TARGET_SAMPLES}",
        "-acodec", "pcm_s16le",
        wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if audio_path != wav_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # Verify sample count after conversion
    try:
        info = sf.info(wav_path)
        if info.frames < TARGET_SAMPLES:
            print(f"WARNING: {wav_path} has {info.frames} samples, expected >= {TARGET_SAMPLES}. Forcing re-pad...")
            _force_pad_wav(wav_path)
    except Exception as e:
        print(f"Could not verify sample count for {wav_path}: {e}")

    return wav_path


def _force_pad_wav(wav_path: str):
    """
    Nuclear option: if ffmpeg apad still produced a short file,
    read the wav with soundfile and zero-pad to TARGET_SAMPLES directly.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="int16", always_2d=False)
    current_samples = len(data)

    if current_samples < TARGET_SAMPLES:
        pad_needed = TARGET_SAMPLES - current_samples
        data = np.concatenate([data, np.zeros(pad_needed, dtype=np.int16)])
        sf.write(wav_path, data, sr, subtype="PCM_16")
        print(f"Force-padded {wav_path}: {current_samples} -> {TARGET_SAMPLES} samples")


def calculate_chunk_size(file_size: int) -> int:
    if file_size <= 0:
        return 1024 * 1024
    if file_size <= 1024 * 1024:
        return 1024
    if file_size <= 1024 * 1024 * 1024:
        return 1024 * 1024
    return 1024 * 1024 * 10


def extract_disposition_params(content_disposition: str) -> Dict[str, str]:
    parts = (p.strip() for p in content_disposition.split(";"))
    params = {
        key.strip().lower(): value.strip().strip('"')
        for part in parts
        if "=" in part
        for key, value in [part.split("=", 1)]
    }
    return params


def _get_extension_from_content_type(content_type: str) -> str:
    if not content_type:
        return ""
    main_type = content_type.split(";")[0].strip().lower()
    ext = CONTENT_TYPE_TO_EXT.get(main_type)
    if ext:
        return ext
    if main_type.startswith("audio/"):
        return ".wav"
    return ""


def download_files_from_urls(job_id: str, urls: Union[str, List[str]]) -> List[str]:
    """
    Accepts a single URL or list of URLs, downloads and normalizes them.
    Returns list of absolute paths to normalized WAV files.
    """
    download_directory = os.path.abspath(os.path.join("jobs", job_id, "downloaded_files"))
    os.makedirs(download_directory, exist_ok=True)

    @backoff.on_exception(backoff.expo, RequestException, max_tries=3)
    def download_file(url: str, path_to_save: str) -> str:
        with requests.Session().get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()

            content_disposition = response.headers.get("Content-Disposition")
            file_extension = ""

            if content_disposition:
                params = extract_disposition_params(content_disposition)
                file_extension = os.path.splitext(params.get("filename", ""))[1]

            if not file_extension:
                file_extension = os.path.splitext(urlparse(url).path)[1]

            if not file_extension:
                file_extension = _get_extension_from_content_type(
                    response.headers.get("Content-Type", "")
                )

            if not file_extension:
                file_extension = ".bin"

            file_size = int(response.headers.get("Content-Length", 0))
            chunk_size = calculate_chunk_size(file_size)

            with open(path_to_save + file_extension, "wb") as f:
                bytes_written = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        if file_size > 0 and bytes_written >= file_size:
                            break

            return file_extension

    def download_file_to_path(url: str) -> str:
        if url is None:
            return None

        file_name = str(uuid.uuid4())
        output_file_path = os.path.join(download_directory, file_name)

        try:
            print("Downloading...", flush=True)
            file_extension = download_file(url, output_file_path)
            print(f"Download complete ({file_extension})", flush=True)
        except RequestException as err:
            print(f"Failed to download {url}: {err}", flush=True)
            return None

        final_path = os.path.abspath(f"{output_file_path}{file_extension}")

        if file_extension.lower() in AUDIO_EXTENSIONS:
            try:
                print("Normalizing with ffmpeg (48kHz mono, exact 480000 sample pad)...", flush=True)
                final_path = normalize_to_wav(final_path)
                print("Normalization complete", flush=True)
            except subprocess.CalledProcessError as err:
                print(f"ffmpeg normalization failed for {final_path}: {err.stderr.decode()}", flush=True)
                return None

        return final_path

    if isinstance(urls, str):
        urls = [urls]

    with ThreadPoolExecutor() as executor:
        downloaded_files = list(executor.map(download_file_to_path, urls))

    return downloaded_files