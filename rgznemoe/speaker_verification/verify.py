import argparse
from pathlib import Path

import torch
import torchaudio

from .config import MODEL_PATH, TARGET_SAMPLE_RATE, CLIP_DURATION_SECONDS
from .model import create_model
from .audio_utils import load_wav


def load_and_prepare(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".wav":
        waveform = load_wav(path, TARGET_SAMPLE_RATE)
    else:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)

    clip_samples = int(CLIP_DURATION_SECONDS * TARGET_SAMPLE_RATE)
    num_samples = waveform.shape[1]
    if num_samples < clip_samples:
        pad = clip_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif num_samples > clip_samples:
        start = torch.randint(0, num_samples - clip_samples + 1, (1,)).item()
        waveform = waveform[:, start : start + clip_samples]

    return waveform.unsqueeze(0)  # (1, 1, T)


def verify_speakers(
    ref_audio: Path,
    test_audio: Path,
    model_path: Path = MODEL_PATH,
) -> None:
    model, device = create_model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        ref_wave = load_and_prepare(ref_audio).to(device)
        test_wave = load_and_prepare(test_audio).to(device)

        ref_emb = model(ref_wave)
        test_emb = model(test_wave)

        sim = torch.nn.functional.cosine_similarity(ref_emb, test_emb).item()

    print(f"Косинусное сходство эмбеддингов: {sim:.3f}")
    if sim > 0.6:
        print("Вероятно, что это один и тот же диктор.")
    else:
        print("Вероятно, что это разные дикторы.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Верификация диктора по двум аудиозаписям")
    parser.add_argument("ref", type=str, help="Путь к опорному аудио (известный диктор)")
    parser.add_argument("test", type=str, help="Путь к тестовому аудио")
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH))

    args = parser.parse_args()

    verify_speakers(Path(args.ref), Path(args.test), Path(args.model_path))


if __name__ == "__main__":
    main()
