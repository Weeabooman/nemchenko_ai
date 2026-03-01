from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Путь к датасету (ожидается структура: data_root/speaker_id/*.wav)
DATA_ROOT = BASE_DIR / "data" / "test1"

# Путь для сохранения обученной модели
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "speaker_embedding_cnn.pt"

# Аудиопараметры
TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION_SECONDS = 3.0  # длина сэмпла для обучения/верификации

# Спектрограммы
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Модель
EMBEDDING_DIM = 128

# Обучение
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
MARGIN = 0.5  # для triplet loss

