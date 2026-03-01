"""
Утилиты загрузки аудио без зависимости от FFmpeg/torchcodec.
Использует scipy для WAV-файлов, чтобы обойти проблемы torchaudio.load на Windows.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    import scipy.io.wavfile as wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_wav(path: Path, target_sr: int) -> torch.Tensor:
    """
    Загружает WAV-файл и приводит к target_sr.
    Возвращает waveform формы (1, T).
    """
    if not HAS_SCIPY:
        raise ImportError("Установите scipy: pip install scipy")

    sr, data = wavfile.read(str(path))
    if data.dtype in (np.int16, np.int32):
        data = data.astype(np.float32) / (
            32768.0 if data.dtype == np.int16 else 2147483648.0
        )
    if data.ndim == 1:
        waveform = torch.from_numpy(data).float().unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.mean(axis=1)).float().unsqueeze(0)

    if sr != target_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform  # (1, T)
