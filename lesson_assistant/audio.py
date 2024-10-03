import logging
from pathlib import Path

from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_audio(video_file: Path, output_file: Path) -> Path:
    try:
        logging.info(f"Extraindo áudio de {video_file} para mp3...")
        output_audio = AudioSegment.from_file(video_file)
        return output_audio.export(output_file, format="mp3")
    except Exception as e:
        logging.error(f"Erro ao extrair o áudio de {video_file}: {e}")
