import os
import json
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from pandas import DataFrame
from pydub import AudioSegment
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import tempfile

Segment = Tuple[float, float, str]


def rttm_to_dataframe(rttm_content: str) -> pd.DataFrame:
    def format_timestamp(seconds: float) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    rows = []

    for line in rttm_content.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 8:
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker_raw = parts[7]

            try:
                spk_idx = int(speaker_raw.split("_")[-1])
                label = chr(65 + spk_idx)
            except (ValueError, IndexError):
                label = speaker_raw

            rows.append(
                {
                    "segment": f"[ {format_timestamp(start)} --> {format_timestamp(end)}]",
                    "label": label,
                    "speaker": speaker_raw.upper(),
                    "start": round(start, 6),
                    "end": round(end, 6),
                }
            )

    return pd.DataFrame(rows)


class NemoDiarization:
    def __init__(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmpdir.name)

        default_config = (
            Path(__file__).resolve().parent
            / "nemo_config"
            / "diar_infer_general.yaml"
        )

        config_path = os.getenv("CUSTOM_NVIDIA_NEMO_CONFIG", str(default_config))
        self.config = OmegaConf.load(config_path)

    def _configure_model(self, manifest_path: Path) -> None:
        diarizer_cfg = self.config.diarizer
        diarizer_cfg.manifest_filepath = str(manifest_path)
        diarizer_cfg.vad.model_path = os.getenv(
            "CUSTOM_VAD_MODEL_PATH", "vad_multilingual_marblenet"
        )
        diarizer_cfg.speaker_embeddings.model_path = os.getenv(
            "CUSTOM_SPEAKER_EMBEDDINGS_MODEL_PATH", "titanet_large"
        )
        diarizer_cfg.out_dir = str(self.output_dir)

    def __call__(self, **extra_fields: Any) -> DataFrame:
        """
        Expected minimum extra_fields:
        - audio_filepath (required by NeMo)
        """

        if "audio_filepath" not in extra_fields:
            raise ValueError("audio_filepath is required in manifest.")

        audio_path = Path(extra_fields["audio_filepath"]).resolve()

        manifest_path = self._create_manifest(**extra_fields)
        self._configure_model(manifest_path)

        model = ClusteringDiarizer(cfg=self.config)
        model.diarize()

        rttm_content = self._get_generated_rttm_path(audio_path).read_text()
        return rttm_to_dataframe(rttm_content)

    def _create_manifest(self, **extra_fields: Any) -> Path:

        manifest_data = {
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "uem_filepath": None,
        }

        manifest_data.update(extra_fields)
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        return manifest_path

    def _get_generated_rttm_path(self, wav_path: Path) -> Path:
        rttm_file = (
            self.output_dir / "pred_rttms" / wav_path.with_suffix(".rttm").name
        )

        if not rttm_file.exists():
            raise FileNotFoundError("No RTTM file generated.")

        return rttm_file

    def __del__(self):
        self._tmpdir.cleanup()
