import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    DATA_ROOT,
    MODEL_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MARGIN,
)
from .audio_dataset import SpeakerAudioDataset
from .model import create_model


def generate_batch_triplets(
    embeddings: torch.Tensor,
    speakers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Простая online-генерация triplet-условий внутри батча.
    Для каждого диктора в батче выбираем несколько (anchor, positive, negative).
    """
    anchors = []
    positives = []
    negatives = []

    speakers_np = speakers.cpu().numpy()
    unique_speakers = list(set(speakers_np))

    for sp in unique_speakers:
        idxs = (speakers_np == sp).nonzero()[0]
        if len(idxs) < 2:
            continue  # нужно минимум два примера одного диктора

        neg_idxs = (speakers_np != sp).nonzero()[0]
        if len(neg_idxs) == 0:
            continue

        for i in range(len(idxs) - 1):
            a_idx = idxs[i]
            p_idx = idxs[i + 1]
            n_idx = int(neg_idxs[torch.randint(0, len(neg_idxs), (1,)).item()])

            anchors.append(embeddings[a_idx])
            positives.append(embeddings[p_idx])
            negatives.append(embeddings[n_idx])

    if not anchors:
        return None, None, None

    return (
        torch.stack(anchors, dim=0),
        torch.stack(positives, dim=0),
        torch.stack(negatives, dim=0),
    )


def train(
    data_root: Path = DATA_ROOT,
    model_path: Path = MODEL_PATH,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    margin: float = MARGIN,
) -> None:
    dataset = SpeakerAudioDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model, device = create_model()
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_triplets = 0

        for waveforms, speaker_idx in loader:
            waveforms = waveforms.to(device)  # (B, 1, T)
            speaker_idx = speaker_idx.to(device)

            optimizer.zero_grad()
            embeddings = model(waveforms)  # (B, D)

            anchor, positive, negative = generate_batch_triplets(embeddings, speaker_idx)
            if anchor is None:
                continue

            loss = criterion(anchor, positive, negative)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * anchor.size(0)
            num_triplets += anchor.size(0)

        if num_triplets > 0:
            avg_loss = epoch_loss / num_triplets
        else:
            avg_loss = 0.0

        print(f"Epoch {epoch}/{num_epochs} - avg triplet loss: {avg_loss:.4f} (triplets={num_triplets})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели идентификации диктора (triplet loss)")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Путь к датасету test1")
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH), help="Куда сохранить модель")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--margin", type=float, default=MARGIN)

    args = parser.parse_args()

    train(
        data_root=Path(args.data_root),
        model_path=Path(args.model_path),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
