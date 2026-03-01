from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
import torchaudio

from .config import TARGET_SAMPLE_RATE, CLIP_DURATION_SECONDS
from .audio_utils import load_wav, HAS_SCIPY


class SpeakerAudioDataset(Dataset):
    """
    Датасет для задач идентификации/верификации диктора.

    Ожидаемая структура папок:

    data_root/
        speaker_1/
            file1.wav
            file2.wav
            ...
        speaker_2/
            file3.wav
            ...

    Каждый подкаталог соответствует отдельному диктору.
    """

    def __init__(self, data_root: Path, extensions: Tuple[str, ...] = (".wav", ".flac", ".mp3")) -> None:
        self.data_root = Path(data_root)
        self.extensions = extensions

        self.sample_rate = TARGET_SAMPLE_RATE
        self.clip_samples = int(CLIP_DURATION_SECONDS * TARGET_SAMPLE_RATE)

        self.items: List[Tuple[Path, int]] = []
        self.speaker_to_idx: Dict[str, int] = {}

        self._index_dataset()

    def _index_dataset(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(f"DATA_ROOT не найден: {self.data_root}. "
                                    f"Ожидается структура data_root/speaker_id/*.wav")

        speaker_dirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        speaker_dirs.sort()

        for sp_idx, sp_dir in enumerate(speaker_dirs):
            self.speaker_to_idx[sp_dir.name] = sp_idx
            for audio_path in sp_dir.rglob("*"):
                if audio_path.suffix.lower() in self.extensions:
                    self.items.append((audio_path, sp_idx))

        if not self.items:
            raise RuntimeError(f"В каталоге {self.data_root} не найдено аудиофайлов "
                               f"c расширениями {self.extensions}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".wav" and HAS_SCIPY:
            waveform = load_wav(path, self.sample_rate)
        else:
            waveform, sr = torchaudio.load(str(path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # приведение к фиксированной длине
        num_samples = waveform.shape[1]
        if num_samples < self.clip_samples:
            pad = self.clip_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif num_samples > self.clip_samples:
            start = torch.randint(0, num_samples - self.clip_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.clip_samples]

        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path, speaker_idx = self.items[idx]
        waveform = self._load_audio(audio_path)
        return waveform, speaker_idx

