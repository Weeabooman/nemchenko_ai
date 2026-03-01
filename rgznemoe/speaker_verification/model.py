from typing import Tuple

import torch
import torch.nn as nn
import torchaudio

from .config import N_MELS, N_FFT, HOP_LENGTH, TARGET_SAMPLE_RATE, EMBEDDING_DIM


class MelSpecExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.ampl_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        :param waveform: (B, 1, T)
        :return: (B, 1, N_MELS, T_frames)
        """
        x = self.melspec(waveform)
        x = self.ampl_to_db(x)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + 1e-6)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x


class SpeakerEmbeddingCNN(nn.Module):
    """
    Простая сверточная сеть, принимающая log-mel спектрограммы и возвращающая L2-нормированные эмбеддинги.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.spec_extractor = MelSpecExtractor()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        :param waveform: (B, 1, T)
        :return: (B, embedding_dim) L2-нормированный эмбеддинг
        """
        x = self.spec_extractor(waveform)
        x = self.cnn(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = self.fc(x)  # (B, embedding_dim)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def create_model(device: torch.device | None = None) -> Tuple[SpeakerEmbeddingCNN, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerEmbeddingCNN()
    model.to(device)
    return model, device

