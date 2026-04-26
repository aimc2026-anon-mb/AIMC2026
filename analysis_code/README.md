# Analysis Code

This folder contains the scripts used to generate and audit the companion-page
audio analysis. It intentionally excludes generated audio, embeddings, figures,
audit folders, caches, and local virtual environments.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Metadata and Embeddings

```bash
python extract_metadata.py --audio-dir ../docs/audit/audio --out embeddings/metadata.csv
python extract_embeddings.py
```

`extract_embeddings.py` and `extract_metadata.py` use `../docs/audit/audio`
as the default audio folder; set `AUDIO_DIR` or pass `--audio-dir` if you want
to analyze a different local FLAC batch.

## Prompt/Denoise Audit

Copy `prompt_denoise_audit.example.json`, replace the ComfyUI output path and
local reproducibility placeholders, then run:

```bash
python prompt_denoise_audit.py --config prompt_denoise_audit.example.json
```

Use `--dry-run` to write the manifest without queueing jobs, or
`--analysis-only` to analyze an existing local audit folder.
