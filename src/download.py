"""
Local download utility for audio URLs.
More robust than runpod's rp_download for URLs like files.pyannote.ai:
- Longer timeouts (5s is too short for ~14MB files)
- Content-Type fallback for extension when URL has none
- All audio is normalized to 16kHz mono WAV via ffmpeg (fixes pyannote sample mismatch)
"""

import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Dict
from urllib.parse import urlparse

import backoff
import requests
from requests import RequestException

HEADERS = {"User-Agent": "runpod-python/0.0.0 (https://runpod.io; support@runpod.io)"}

# Content-Type -> extension mapping for when URL has no extension
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

# Timeout: (connect, read) - 30s to connect, 180s to read (for large files)
TIMEOUT = (30, 180)

# Normalized WAV: 16kHz mono 16-bit (required for pyannote/Qwen, avoids sample mismatch)
WAV_SAMPLE_RATE = 16000
WAV_CHANNELS = 1

# Formats that need conversion (everything except already-normalized 16kHz mono WAV)
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".webm", ".flac", ".wav"}


def normalize_to_wav(audio_path: str) -> str:
    """
    Convert any audio to 16kHz mono 16-bit WAV via ffmpeg.
    Ensures consistent format for Qwen ASR and pyannote diarization
    (fixes "expected 160000 samples" ValueError).
    Returns path to the .wav file. Removes original if different.
    """
    base, ext = os.path.splitext(audio_path)
    ext_lower = ext.lower()
    wav_path = base + ".wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ar", str(WAV_SAMPLE_RATE),
        "-ac", str(WAV_CHANNELS),
        "-acodec", "pcm_s16le",
        "-fflags", "+genpts",  # better duration alignment for variable formats
        wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if audio_path != wav_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass
    return wav_path


def calculate_chunk_size(file_size: int) -> int:
    if file_size <= 0:
        return 1024 * 1024  # 1 MB default when size unknown
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
    """Extract extension from Content-Type header (e.g. 'audio/x-wav' -> '.wav')."""
    if not content_type:
        return ""
    # Handle "audio/x-wav; charset=..." etc.
    main_type = content_type.split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(main_type, "")


def download_files_from_urls(job_id: str, urls: Union[str, List[str]]) -> List[str]:
    """
    Accepts a single URL or a list of URLs and downloads the files.
    Returns the list of downloaded file absolute paths.
    Saves files in jobs/<job_id>/downloaded_files/.
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
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            return file_extension

    def download_file_to_path(url: str) -> str:
        if url is None:
            return None
        file_name = str(uuid.uuid4())
        output_file_path = os.path.join(download_directory, file_name)
        try:
            file_extension = download_file(url, output_file_path)
        except RequestException as err:
            print(f"Failed to download {url}: {err}")
            return None
        final_path = os.path.abspath(f"{output_file_path}{file_extension}")
        # Normalize all audio to 16kHz mono WAV (fixes pyannote sample mismatch)
        if file_extension.lower() in AUDIO_EXTENSIONS:
            try:
                final_path = normalize_to_wav(final_path)
            except subprocess.CalledProcessError as err:
                print(f"ffmpeg normalization failed for {final_path}: {err}")
        return final_path

    if isinstance(urls, str):
        urls = [urls]

    with ThreadPoolExecutor() as executor:
        downloaded_files = list(executor.map(download_file_to_path, urls))

    return downloaded_files
