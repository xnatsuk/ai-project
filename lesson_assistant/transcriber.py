import logging
from pathlib import Path

from whisperplus import SpeechToTextPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Transcriber:
    def __init__(
            self,
            model,
            hqq_config,
            audio_file: Path,
            output_dir: Path,
    ):
        self.model = model
        self.hqq_config = hqq_config
        self.audio_file = audio_file
        self.output_dir = output_dir

    def transcribe(self):
        pipeline = SpeechToTextPipeline(
            model_id=self.model, quant_config=self.hqq_config, flash_attention_2=False
        )
        transcript = pipeline(
            audio_path=str(self.audio_file),
            chunk_length_s=30,
            max_new_tokens=128,
            stride_length_s=5,
            batch_size=100,
            language="portuguese",
        )
        if not self.audio_file.exists() or self.audio_file.suffix != ".mp3":
            raise FileNotFoundError("Arquivo de áudio não foi encontrado.")
        try:
            output_text = self.output_dir.joinpath(
                f"{self.audio_file.stem}_transcription"
            ).with_suffix(".txt")
            return self.write_to_file(transcript, output_text)
        except Exception as e:
            logging.error(f"Error: {e}")

    @staticmethod
    def write_to_file(segment, output_text: Path):
        with open(output_text, "w") as f:
            f.write(segment["text"] + "\n")
