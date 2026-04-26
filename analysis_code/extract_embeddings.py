"""
Extract CLAP audio embeddings from the companion-page audio batches.

Reads FLAC files from AUDIO_DIR, runs each through LAION-CLAP's audio encoder,
and writes a single .npz containing the 512-dim embeddings, filenames, and
regime labels (parsed from the leading token of each filename, e.g. "E1" from
"E1_a.flac").

Run with:
    python extract_embeddings.py

Edit AUDIO_DIR at the top of the file if your audio lives elsewhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE.parent / "docs" / "audit" / "audio"
OUTPUT_PATH = HERE / "embeddings" / "clap_embeddings.npz"
MODEL_ID = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000  # CLAP's expected input rate
# ---------------------------------------------------------------------------


def regime_from_filename(path: Path) -> str:
    """Parse regime label from filename, e.g. 'E1_a.flac' -> 'E1'."""
    match = re.match(r"^([A-Za-z]+\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse regime from {path.name}")
    return match.group(1)


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Load audio file, downmix to mono, resample to target sr."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLAP ({MODEL_ID}) on {device}...")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)

    audio_files = sorted(AUDIO_DIR.glob("*.flac"))
    if not audio_files:
        raise FileNotFoundError(f"No .flac files in {AUDIO_DIR}")
    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")

    embeddings: list[np.ndarray] = []
    filenames: list[str] = []
    regimes: list[str] = []

    with torch.no_grad():
        for path in audio_files:
            audio = load_audio(path, SAMPLE_RATE)
            inputs = processor(
                audio=audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # API drift: older transformers returned a projected tensor from
            # get_audio_features; newer versions return BaseModelOutputWithPooling
            # where pooler_output is already the projected audio embedding.
            out = model.get_audio_features(**inputs)
            if isinstance(out, torch.Tensor):
                embed_tensor = out
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                embed_tensor = out.pooler_output
            else:
                # Last-resort fallback: mean-pool last_hidden_state and project
                pooled = out.last_hidden_state.mean(dim=1)
                embed_tensor = model.audio_projection(pooled)

            embed = embed_tensor.squeeze(0).cpu().numpy()  # (512,)

            embeddings.append(embed)
            filenames.append(path.name)
            regimes.append(regime_from_filename(path))
            print(f"  {path.name:20s} -> {regimes[-1]}  (norm={np.linalg.norm(embed):.3f})")

    embeddings_arr = np.stack(embeddings)  # (N, 512)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        embeddings=embeddings_arr,
        filenames=np.array(filenames),
        regimes=np.array(regimes),
    )
    print(f"\nSaved {embeddings_arr.shape} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
