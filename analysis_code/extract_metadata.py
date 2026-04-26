"""
Extract generation metadata from ComfyUI-produced FLAC files.

ComfyUI embeds the full prompt graph (API format) into each output FLAC as a
VORBIS comment. This script reads that graph, finds the sampler node and its
connected positive prompt, and writes a CSV with one row per file:

    filename, regime, variant, prompt, cfg, steps, denoise, sampler, scheduler, seed

Filename convention (hyphen- and number-tolerant):

    E1_a.flac            -> regime=E1, variant="",       index=a
    E1_center_0001.flac  -> regime=E1, variant=center,   index=0001
    E1_cfg06_0001.flac   -> regime=E1, variant=cfg06,    index=0001
    CTRL_piano_0001.flac -> regime=CTRL, variant=piano,  index=0001

Run with:

    python extract_metadata.py                                # default audio folder
    python extract_metadata.py --audio-dir path/to/audio_v2   # custom folder
    python extract_metadata.py --out custom_metadata.csv      # custom CSV output
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_AUDIO_DIR = HERE.parent / "docs" / "audit" / "audio"
DEFAULT_OUT_CSV = HERE / "embeddings" / "metadata.csv"

# Sampler-like node classes we know about. If you use an unusual sampler node,
# add its class_type here.
SAMPLER_CLASSES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
}


def read_vorbis_prompt(flac_path: Path) -> dict | None:
    """Read the embedded ComfyUI prompt graph from a FLAC file's VORBIS comment."""
    with flac_path.open("rb") as f:
        if f.read(4) != b"fLaC":
            return None
        while True:
            header = f.read(4)
            if len(header) < 4:
                return None
            last = (header[0] & 0x80) != 0
            btype = header[0] & 0x7F
            blen = (header[1] << 16) | (header[2] << 8) | header[3]
            if btype == 4:  # VORBIS_COMMENT
                body = f.read(blen)
                vendor_len = struct.unpack("<I", body[:4])[0]
                offset = 4 + vendor_len
                num_comments = struct.unpack("<I", body[offset:offset + 4])[0]
                offset += 4
                for _ in range(num_comments):
                    clen = struct.unpack("<I", body[offset:offset + 4])[0]
                    offset += 4
                    comment = body[offset:offset + clen].decode("utf-8", errors="replace")
                    offset += clen
                    key, _, val = comment.partition("=")
                    if key == "prompt":
                        try:
                            return json.loads(val)
                        except json.JSONDecodeError:
                            return None
                return None
            else:
                f.read(blen)
            if last:
                return None


def find_sampler_node(graph: dict) -> tuple[str, dict] | None:
    """Return (node_id, node_dict) for the first sampler-like node found."""
    for nid, node in graph.items():
        if node.get("class_type") in SAMPLER_CLASSES:
            return nid, node
    return None


def follow_text(graph: dict, ref: list) -> str:
    """Follow an input reference [node_id, output_idx] to a CLIPTextEncode and
    return its text. Returns '' if no text found."""
    if not isinstance(ref, list) or len(ref) < 1:
        return ""
    node = graph.get(str(ref[0]))
    if node is None:
        return ""
    if node.get("class_type") == "CLIPTextEncode":
        return str(node.get("inputs", {}).get("text", ""))
    # Some workflows wrap the text encoder in a conditioning-combine or similar;
    # walk the first input we find if we landed somewhere unexpected.
    for v in node.get("inputs", {}).values():
        if isinstance(v, list) and len(v) >= 1:
            return follow_text(graph, v)
    return ""


def extract_sampler_params(node: dict) -> dict:
    """Normalize sampler params across KSampler and *Advanced variants."""
    ins = node.get("inputs", {})
    return {
        "seed": ins.get("seed", ins.get("noise_seed", "")),
        "steps": ins.get("steps", ""),
        "cfg": ins.get("cfg", ""),
        "sampler": ins.get("sampler_name", ""),
        "scheduler": ins.get("scheduler", ""),
        "denoise": ins.get("denoise", ""),
    }


FILENAME_RE = re.compile(r"^(?P<regime>[A-Za-z]+\d*)(?:_(?P<rest>.+))?$")


def parse_filename(stem: str) -> tuple[str, str, str]:
    """Return (regime, variant, index) parsed from a filename stem."""
    m = FILENAME_RE.match(stem)
    if not m:
        return ("", "", stem)
    regime = m.group("regime")
    rest = m.group("rest") or ""
    if not rest:
        return (regime, "", "")
    parts = rest.split("_")
    if len(parts) == 1:
        # Single trailing token -> index (e.g. E1_a -> regime=E1, index=a)
        return (regime, "", parts[0])
    # Multiple tokens -> everything but last is variant, last is index
    return (regime, "_".join(parts[:-1]), parts[-1])


def extract_row(flac_path: Path) -> dict | None:
    graph = read_vorbis_prompt(flac_path)
    if graph is None:
        print(f"  SKIP (no prompt metadata): {flac_path.name}")
        return None

    sampler = find_sampler_node(graph)
    if sampler is None:
        print(f"  SKIP (no sampler node found): {flac_path.name}")
        return None

    _, sampler_node = sampler
    params = extract_sampler_params(sampler_node)
    positive_ref = sampler_node.get("inputs", {}).get("positive")
    prompt_text = follow_text(graph, positive_ref) if positive_ref else ""

    regime, variant, index = parse_filename(flac_path.stem)
    return {
        "filename": flac_path.name,
        "regime": regime,
        "variant": variant,
        "index": index,
        "prompt": prompt_text,
        "cfg": params["cfg"],
        "steps": params["steps"],
        "denoise": params["denoise"],
        "sampler": params["sampler"],
        "scheduler": params["scheduler"],
        "seed": params["seed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    audio_files = sorted(args.audio_dir.glob("*.flac"))
    if not audio_files:
        raise SystemExit(f"No .flac files in {args.audio_dir}")
    print(f"Scanning {len(audio_files)} files in {args.audio_dir}")

    rows: list[dict] = []
    for path in audio_files:
        row = extract_row(path)
        if row is not None:
            rows.append(row)
            print(
                f"  {path.name:30s} "
                f"regime={row['regime']:5s} "
                f"variant={row['variant']:10s} "
                f"cfg={row['cfg']:>4}  "
                f"denoise={row['denoise']:>4}  "
                f"seed={row['seed']}"
            )

    if not rows:
        raise SystemExit("No rows extracted. Check that files have embedded workflow metadata.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
