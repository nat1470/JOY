from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return np.clip(audio / peak, -1.0, 1.0)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> Tuple[bytes, int]:
    normalized = normalize_audio(audio)
    with sf.SoundFile("/tmp/joy_temp.wav", "wb", samplerate=sample_rate, channels=1, format="WAV") as f:
        f.write(normalized)
    data, _ = sf.read("/tmp/joy_temp.wav", dtype="int16")
    return data.tobytes(), sample_rate
