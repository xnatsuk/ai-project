import logging
from pathlib import Path

import torch
from transformers import HqqConfig, BitsAndBytesConfig

from audio import extract_audio
from transcriber import Transcriber

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Diretórios base e de saída
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arquivo de vídeo e áudio
video_file = BASE_DIR.parent / "assets/aula.mp4"
audio_output = OUTPUT_DIR / "aula.mp3"
output_txt = OUTPUT_DIR / "aula_transcription.txt"

hqq_config = HqqConfig(
    nbits=4,
    group_size=64,
    quant_zero=False,
    quant_scale=False,
    axis=0,
    offload_meta=False,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = "openai/whisper-small"

transcriber = Transcriber(
    model=model,
    hqq_config=hqq_config,
    output_dir=OUTPUT_DIR,
    audio_file=audio_output,
)
if __name__ == "__main__":
    try:
        if not audio_output.exists():
            audio_output = extract_audio(video_file, audio_output)

        transcriber.transcribe()
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")
