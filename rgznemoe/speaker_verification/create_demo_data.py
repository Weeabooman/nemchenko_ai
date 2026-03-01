"""
Создание демо-датасета для тестирования пайплайна обучения.
Генерирует синтетические WAV-файлы (шум + синусоида) для нескольких «дикторов».
Результат не отражает реальную речь, но позволяет запустить обучение без VoxCeleb.
"""
from pathlib import Path
import numpy as np

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

from .config import DATA_ROOT, TARGET_SAMPLE_RATE, CLIP_DURATION_SECONDS


def create_demo_dataset(
    data_root: Path = DATA_ROOT,
    num_speakers: int = 5,
    num_files_per_speaker: int = 10,
    duration_sec: float = None,
) -> None:
    if duration_sec is None:
        duration_sec = CLIP_DURATION_SECONDS

    if wavfile is None:
        raise ImportError("Установите scipy: pip install scipy")

    data_root.mkdir(parents=True, exist_ok=True)
    sr = TARGET_SAMPLE_RATE
    n_samples = int(duration_sec * sr)

    for sp in range(1, num_speakers + 1):
        sp_dir = data_root / f"speaker_{sp:02d}"
        sp_dir.mkdir(exist_ok=True)

        # У каждого «диктора» — свой «голос» (базовая частота синуса)
        base_freq = 200 + sp * 80 + np.random.uniform(-20, 20)

        for f in range(1, num_files_per_speaker + 1):
            t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
            signal = 0.3 * np.sin(2 * np.pi * base_freq * t)
            signal += 0.1 * np.random.randn(n_samples).astype(np.float32)
            signal = np.clip(signal, -1.0, 1.0)
            signal_int16 = (signal * 32767).astype(np.int16)

            out_path = sp_dir / f"{f:04d}.wav"
            wavfile.write(str(out_path), sr, signal_int16)

    print(f"Создан демо-датасет: {data_root}")
    print(f"  Дикторов: {num_speakers}, файлов на диктора: {num_files_per_speaker}")


if __name__ == "__main__":
    create_demo_dataset()
