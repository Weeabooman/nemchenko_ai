import gradio as gr
import torch
import torchaudio

from .config import MODEL_PATH, TARGET_SAMPLE_RATE, CLIP_DURATION_SECONDS
from .model import create_model


model, device = create_model()
if MODEL_PATH.exists():
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
model.eval()


def _load_and_prepare_bytes(audio_tuple):
    if audio_tuple is None:
        return None
    sample_rate, data = audio_tuple
    waveform = torch.tensor(data).float().t()  # (T, C) -> (C, T)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

    clip_samples = int(CLIP_DURATION_SECONDS * TARGET_SAMPLE_RATE)
    num_samples = waveform.shape[1]
    if num_samples < clip_samples:
        pad = clip_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif num_samples > clip_samples:
        start = torch.randint(0, num_samples - clip_samples + 1, (1,)).item()
        waveform = waveform[:, start : start + clip_samples]

    return waveform.unsqueeze(0)  # (1, 1, T)


def verify_gradio(ref_audio, test_audio):
    if ref_audio is None or test_audio is None:
        return "Загрузите обе аудиозаписи."

    with torch.no_grad():
        ref_wave = _load_and_prepare_bytes(ref_audio).to(device)
        test_wave = _load_and_prepare_bytes(test_audio).to(device)

        ref_emb = model(ref_wave)
        test_emb = model(test_wave)

        sim = torch.nn.functional.cosine_similarity(ref_emb, test_emb).item()

    verdict = "один и тот же диктор" if sim > 0.6 else "разные дикторы"
    return f"Косинусное сходство: {sim:.3f}. Предположительно, это {verdict}."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Идентификация диктора (one-shot)") as demo:
        gr.Markdown(
            "## Идентификация диктора (one-shot learning)\n"
            "Загрузите две короткие аудиозаписи для проверки, один ли это диктор."
        )

        with gr.Row():
            ref_audio = gr.Audio(label="Опорная запись", type="numpy")
            test_audio = gr.Audio(label="Тестовая запись", type="numpy")

        btn = gr.Button("Проверить")
        output = gr.Textbox(label="Результат")

        btn.click(fn=verify_gradio, inputs=[ref_audio, test_audio], outputs=output)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()

