"""
End-to-end matched audit pipeline for ComfyUI prompt/denoise studies.

This script is designed for the paper's controlled analysis batch rather than
for the artistic exemplar table. It:

1. Loads a saved ComfyUI workflow in API prompt format.
2. Builds the Cartesian product of prompt families x prompts x denoise values
   x matched seed slots.
3. Queues every render job via ComfyUI's HTTP prompt API.
4. Waits for the resulting FLAC files, copies them into a local audit folder,
   and verifies the embedded workflow metadata.
5. Computes CLAP audio embeddings and writes:
   - within-family dispersion
   - distance to the unconditioned centroid
   - audio-audio separation by denoise level
   - PCA seed-matched trajectories
6. Produces JSON/CSV summaries plus PNG/PDF figures.

Run:
    python prompt_denoise_audit.py --config prompt_denoise_audit.example.json

Useful flags:
    --dry-run        Build manifest only; do not queue or analyze.
    --queue-only     Queue and harvest renders, but do not analyze.
    --analysis-only  Skip ComfyUI and analyze existing local audit audio.
    --rerun-existing Requeue jobs even if local copies already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from transformers import ClapModel, ClapProcessor

from extract_metadata import (
    extract_sampler_params,
    find_sampler_node,
    follow_text,
    read_vorbis_prompt,
)

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib.pyplot as plt

AUDITS_DIR = HERE / "audits"
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48000

# Paper-friendly defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
})


@dataclass
class JobSpec:
    experiment_name: str
    family_id: str
    family_label: str
    prompt_id: str
    prompt_label: str
    prompt_text: str
    prompt_slot: int
    negative_prompt: str
    seed: int
    seed_slot: int
    denoise: float
    cfg: float
    steps: int
    sampler_name: str
    scheduler: str
    seconds: float | None
    batch_size: int | None
    comfy_prefix: str
    local_filename: str
    comfy_prompt_id: str
    family_order: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {" ", "-", "_"}:
            safe.append("-")
    slug = "".join(safe).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "audit"


def denoise_tag(denoise: float) -> str:
    return f"dn{int(round(denoise * 100)):03d}"


def normalize_prompt_entry(family_id: str, family_label: str, entry: Any, prompt_slot: int) -> dict[str, Any]:
    if isinstance(entry, str):
        prompt_text = entry
        prompt_id = f"{family_id}-p{prompt_slot:02d}"
        prompt_label = prompt_text or f"{family_label} {prompt_slot}"
        return {
            "id": prompt_id,
            "label": prompt_label,
            "prompt": prompt_text,
            "slot": prompt_slot,
        }

    if not isinstance(entry, dict):
        raise ValueError("Prompt entries must be strings or objects")
    if "prompt" not in entry:
        raise ValueError("Prompt entry objects must define 'prompt'")

    prompt_text = str(entry["prompt"])
    prompt_id = slugify(str(entry.get("id", f"{family_id}-p{prompt_slot:02d}")))
    prompt_label = str(entry.get("label", prompt_text or f"{family_label} {prompt_slot}"))
    return {
        "id": prompt_id,
        "label": prompt_label,
        "prompt": prompt_text,
        "slot": int(entry.get("slot", prompt_slot)),
    }


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)

    base_dir = path.resolve().parent
    config["config_path"] = str(path.resolve())
    config["workflow_path"] = str(resolve_path(base_dir, config["workflow_path"]))
    comfyui = config.setdefault("comfyui", {})
    if "output_dir" not in comfyui:
        raise ValueError("Config must define comfyui.output_dir")
    comfyui["output_dir"] = str(resolve_path(base_dir, comfyui["output_dir"]))
    comfyui.setdefault("host", "127.0.0.1:8188")

    node_ids = config.setdefault("node_ids", {})
    required_nodes = {"positive_text", "sampler", "save_audio"}
    missing_nodes = sorted(required_nodes - set(node_ids))
    if missing_nodes:
        raise ValueError(f"Missing node_ids entries: {', '.join(missing_nodes)}")
    for key, value in list(node_ids.items()):
        node_ids[key] = str(value)

    fixed_inputs = config.setdefault("fixed_inputs", {})
    fixed_sampler = fixed_inputs.setdefault("sampler", {})
    for key in ("cfg", "steps", "sampler_name", "scheduler"):
        if key not in fixed_sampler:
            raise ValueError(f"fixed_inputs.sampler must define '{key}'")
    fixed_inputs.setdefault("negative_prompt", "")
    fixed_inputs.setdefault("latent_audio", {})

    prompt_families = config.get("prompt_families", [])
    if not prompt_families:
        raise ValueError("Config must define at least one prompt family")
    for family in prompt_families:
        if "id" not in family:
            raise ValueError("Each prompt family must define 'id'")
        family.setdefault("label", family["id"])
        family.setdefault("sampler_overrides", {})
        family_id = slugify(family["id"])
        family_label = family["label"]
        if "prompts" in family:
            raw_prompts = family["prompts"]
        elif "prompt" in family:
            raw_prompts = [family["prompt"]]
        else:
            raise ValueError("Each prompt family must define either 'prompt' or 'prompts'")
        if not raw_prompts:
            raise ValueError(f"Prompt family '{family['id']}' has no prompts")
        family["prompts"] = [
            normalize_prompt_entry(family_id, family_label, entry, idx + 1)
            for idx, entry in enumerate(raw_prompts)
        ]

    denoise_values = config.get("denoise_values", [])
    seeds = config.get("seeds", [])
    seed_slots = config.get("seed_slots", [])
    if not denoise_values:
        raise ValueError("Config must define denoise_values")
    if seed_slots:
        for idx, slot in enumerate(seed_slots, start=1):
            if not slot:
                raise ValueError(f"seed_slots[{idx}] is empty")
            seed_slots[idx - 1] = [int(seed) for seed in slot]
    elif seeds:
        config["seeds"] = [int(seed) for seed in seeds]
    else:
        raise ValueError("Config must define either seeds or seed_slots")

    max_prompt_count = max(len(family["prompts"]) for family in prompt_families)
    if seed_slots and len(seed_slots) < max_prompt_count:
        raise ValueError(
            f"seed_slots defines {len(seed_slots)} slots but the largest prompt family has "
            f"{max_prompt_count} prompts"
        )

    config.setdefault("experiment_name", path.stem)
    config.setdefault("control_family_id", prompt_families[-1]["id"])
    config.setdefault("save_prefix_root", "analysis_audits")
    config.setdefault("null_samples", 1000)
    return config


def audit_root(config: dict[str, Any]) -> Path:
    return AUDITS_DIR / slugify(config["experiment_name"])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_workflow(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_jobs(config: dict[str, Any]) -> list[JobSpec]:
    fixed_inputs = config["fixed_inputs"]
    fixed_sampler = dict(fixed_inputs["sampler"])
    latent_audio = dict(fixed_inputs.get("latent_audio", {}))
    seconds = latent_audio.get("seconds")
    batch_size = latent_audio.get("batch_size")
    save_prefix_root = Path(config["save_prefix_root"])
    experiment_slug = slugify(config["experiment_name"])

    jobs: list[JobSpec] = []
    for family_order, family in enumerate(config["prompt_families"]):
        family_id = slugify(family["id"])
        family_label = family["label"]
        sampler_overrides = dict(family.get("sampler_overrides", {}))
        negative_prompt = family.get("negative_prompt", fixed_inputs["negative_prompt"])

        for prompt_entry in family["prompts"]:
            prompt_id = slugify(prompt_entry["id"])
            prompt_label = prompt_entry["label"]
            prompt_text = prompt_entry["prompt"]
            prompt_slot = int(prompt_entry["slot"])
            if config.get("seed_slots"):
                seeds_for_prompt = config["seed_slots"][prompt_slot - 1]
            else:
                seeds_for_prompt = config["seeds"]

            for denoise in config["denoise_values"]:
                for seed_slot, seed in enumerate(seeds_for_prompt, start=1):
                    sampler_inputs = dict(fixed_sampler)
                    sampler_inputs.update(sampler_overrides)
                    sampler_inputs["seed"] = int(seed)
                    sampler_inputs["denoise"] = float(denoise)

                    base_name = (
                        f"{family_id}__{prompt_id}__{denoise_tag(float(denoise))}"
                        f"__slot{seed_slot:02d}__seed{int(seed)}"
                    )
                    comfy_prefix = str(
                        save_prefix_root / experiment_slug / family_id / prompt_id / base_name
                    )
                    local_filename = f"{base_name}.flac"
                    jobs.append(JobSpec(
                        experiment_name=config["experiment_name"],
                        family_id=family_id,
                        family_label=family_label,
                        prompt_id=prompt_id,
                        prompt_label=prompt_label,
                        prompt_text=prompt_text,
                        prompt_slot=prompt_slot,
                        negative_prompt=negative_prompt,
                        seed=int(seed),
                        seed_slot=seed_slot,
                        denoise=float(denoise),
                        cfg=float(sampler_inputs["cfg"]),
                        steps=int(sampler_inputs["steps"]),
                        sampler_name=str(sampler_inputs["sampler_name"]),
                        scheduler=str(sampler_inputs["scheduler"]),
                        seconds=float(seconds) if seconds is not None else None,
                        batch_size=int(batch_size) if batch_size is not None else None,
                        comfy_prefix=comfy_prefix,
                        local_filename=local_filename,
                        comfy_prompt_id=str(uuid.uuid4()),
                        family_order=family_order,
                    ))
    return jobs


def build_prompt_payload(
    workflow_template: dict[str, Any],
    config: dict[str, Any],
    job: JobSpec,
) -> dict[str, Any]:
    workflow = deepcopy(workflow_template)
    node_ids = config["node_ids"]
    fixed_inputs = config["fixed_inputs"]

    positive_node = workflow[node_ids["positive_text"]]
    positive_node["inputs"]["text"] = job.prompt_text

    negative_id = node_ids.get("negative_text")
    if negative_id is not None:
        workflow[negative_id]["inputs"]["text"] = job.negative_prompt

    sampler_node = workflow[node_ids["sampler"]]
    sampler_inputs = sampler_node["inputs"]
    sampler_inputs["seed"] = job.seed
    sampler_inputs["steps"] = job.steps
    sampler_inputs["cfg"] = job.cfg
    sampler_inputs["sampler_name"] = job.sampler_name
    sampler_inputs["scheduler"] = job.scheduler
    sampler_inputs["denoise"] = job.denoise

    latent_id = node_ids.get("latent_audio")
    if latent_id is not None:
        latent_inputs = workflow[latent_id]["inputs"]
        for key, value in fixed_inputs.get("latent_audio", {}).items():
            latent_inputs[key] = value

    save_node = workflow[node_ids["save_audio"]]
    save_node["inputs"]["filename_prefix"] = job.comfy_prefix
    return workflow


def comfy_request_json(host: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"http://{host}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body)


def queue_job(host: str, workflow: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    payload = {
        "prompt": workflow,
        "client_id": f"audit-{uuid.uuid4()}",
        "prompt_id": prompt_id,
    }
    return comfy_request_json(host, "/prompt", payload)


def get_history(host: str, prompt_id: str) -> dict[str, Any]:
    try:
        return comfy_request_json(host, f"/history/{prompt_id}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise


def find_rendered_audio(output_dir: Path, comfy_prefix: str) -> Path | None:
    prefix_path = Path(comfy_prefix)
    expected_dir = output_dir / prefix_path.parent
    if not expected_dir.exists():
        return None
    matches = sorted(
        expected_dir.glob(f"{prefix_path.name}*.flac"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def copy_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def extract_actual_metadata(path: Path) -> dict[str, Any]:
    graph = read_vorbis_prompt(path)
    if graph is None:
        return {
            "actual_prompt": "",
            "actual_cfg": "",
            "actual_steps": "",
            "actual_denoise": "",
            "actual_sampler_name": "",
            "actual_scheduler": "",
            "actual_seed": "",
            "metadata_ok": False,
            "metadata_issue": "missing prompt graph",
        }

    sampler = find_sampler_node(graph)
    if sampler is None:
        return {
            "actual_prompt": "",
            "actual_cfg": "",
            "actual_steps": "",
            "actual_denoise": "",
            "actual_sampler_name": "",
            "actual_scheduler": "",
            "actual_seed": "",
            "metadata_ok": False,
            "metadata_issue": "missing sampler node",
        }

    _, sampler_node = sampler
    params = extract_sampler_params(sampler_node)
    positive_ref = sampler_node.get("inputs", {}).get("positive")
    actual_prompt = follow_text(graph, positive_ref) if positive_ref else ""
    return {
        "actual_prompt": actual_prompt,
        "actual_cfg": params["cfg"],
        "actual_steps": params["steps"],
        "actual_denoise": params["denoise"],
        "actual_sampler_name": params["sampler"],
        "actual_scheduler": params["scheduler"],
        "actual_seed": params["seed"],
        "metadata_ok": True,
        "metadata_issue": "",
    }


def compare_expected_actual(job: JobSpec, actual: dict[str, Any]) -> tuple[bool, str]:
    if not actual["metadata_ok"]:
        return False, actual["metadata_issue"]

    mismatches: list[str] = []
    checks = [
        ("prompt", job.prompt_text, str(actual["actual_prompt"])),
        ("cfg", job.cfg, actual["actual_cfg"]),
        ("steps", job.steps, actual["actual_steps"]),
        ("denoise", job.denoise, actual["actual_denoise"]),
        ("sampler_name", job.sampler_name, str(actual["actual_sampler_name"])),
        ("scheduler", job.scheduler, str(actual["actual_scheduler"])),
        ("seed", job.seed, actual["actual_seed"]),
    ]
    for field, expected, observed in checks:
        if str(expected) != str(observed):
            mismatches.append(f"{field}: expected={expected} observed={observed}")
    return (not mismatches), "; ".join(mismatches)


def local_audio_path(experiment_root: Path, job: JobSpec) -> Path:
    return experiment_root / "audio" / job.local_filename


def gather_existing_or_queue(
    config: dict[str, Any],
    jobs: list[JobSpec],
    workflow_template: dict[str, Any],
    experiment_root: Path,
    rerun_existing: bool,
    poll_seconds: float,
    timeout_minutes: float,
) -> list[dict[str, Any]]:
    output_dir = Path(config["comfyui"]["output_dir"])
    host = config["comfyui"]["host"]
    timeout_seconds = timeout_minutes * 60

    manifest_rows: list[dict[str, Any]] = []
    pending: list[JobSpec] = []
    for job in jobs:
        local_path = local_audio_path(experiment_root, job)
        discovered = None if rerun_existing else find_rendered_audio(output_dir, job.comfy_prefix)
        if local_path.exists() and not rerun_existing:
            status = "local_exists"
            source_path = str(local_path)
        elif discovered is not None and not rerun_existing:
            copy_audio(discovered, local_path)
            status = "copied_existing_output"
            source_path = str(discovered)
        else:
            prompt = build_prompt_payload(workflow_template, config, job)
            print(f"Queueing {job.local_filename}")
            try:
                queue_response = queue_job(host, prompt, job.comfy_prompt_id)
            except urllib.error.URLError as exc:
                raise SystemExit(
                    "Could not reach the ComfyUI API at "
                    f"http://{host}. Start ComfyUI with API access first, then rerun."
                ) from exc
            print(f"  prompt_id={queue_response.get('prompt_id', job.comfy_prompt_id)}")
            status = "queued"
            source_path = ""
            pending.append(job)

        row = asdict(job)
        row.update({
            "queued_at_utc": utc_now_iso(),
            "status": status,
            "source_output_path": source_path,
            "local_copy_path": str(local_path),
        })
        manifest_rows.append(row)

    if not pending:
        return manifest_rows

    pending_by_id = {job.comfy_prompt_id: job for job in pending}
    started = time.time()
    while pending_by_id:
        if time.time() - started > timeout_seconds:
            still_pending = ", ".join(job.local_filename for job in pending_by_id.values())
            raise TimeoutError(f"Timed out waiting for: {still_pending}")

        for prompt_id, job in list(pending_by_id.items()):
            local_path = local_audio_path(experiment_root, job)
            rendered = find_rendered_audio(output_dir, job.comfy_prefix)
            history = get_history(host, prompt_id)
            if rendered is None and not history:
                continue

            if rendered is not None:
                copy_audio(rendered, local_path)
                status = "rendered"
                source_path = str(rendered)
            else:
                continue

            for row in manifest_rows:
                if row["comfy_prompt_id"] == prompt_id:
                    row["status"] = status
                    row["source_output_path"] = source_path
                    row["completed_at_utc"] = utc_now_iso()
                    break
            print(f"Harvested {job.local_filename} [{status}]")
            del pending_by_id[prompt_id]

        if pending_by_id:
            time.sleep(poll_seconds)

    return manifest_rows


def resolve_local_clap_snapshot() -> Path | None:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_dir / "models--laion--clap-htsat-unfused" / "snapshots"
    if not repo_dir.exists():
        return None
    candidates = sorted(repo_dir.iterdir(), reverse=True)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_config = (candidate / "config.json").exists()
        has_processor = (candidate / "preprocessor_config.json").exists()
        has_weight = (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists()
        if has_config and has_processor and has_weight:
            return candidate
    return None


def load_clap() -> tuple[ClapModel, ClapProcessor, str]:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"

    model: ClapModel | None = None
    processor: ClapProcessor | None = None
    errors: list[str] = []

    sources: list[tuple[str | Path, dict[str, Any]]] = [
        (CLAP_MODEL_ID, {"local_files_only": True}),
    ]
    local_snapshot = resolve_local_clap_snapshot()
    if local_snapshot is not None:
        sources.append((local_snapshot, {"local_files_only": True}))
    sources.append((CLAP_MODEL_ID, {}))

    for source, kwargs in sources:
        try:
            model = ClapModel.from_pretrained(str(source), **kwargs).to(device).eval()
            processor = ClapProcessor.from_pretrained(str(source), **kwargs)
            break
        except OSError as exc:
            errors.append(f"{source}: {exc}")

    if model is None or processor is None:
        joined = "\n".join(errors)
        raise OSError(f"Could not load CLAP model.\n{joined}")

    return model, processor, device


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def embed_audio_files(
    audio_paths: list[Path],
    model: ClapModel,
    processor: ClapProcessor,
    device: str,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    print(f"Embedding {len(audio_paths)} files with CLAP on {device}...")
    with torch.no_grad():
        for path in audio_paths:
            audio = load_audio(path)
            inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            out = model.get_audio_features(**inputs)
            if isinstance(out, torch.Tensor):
                embed_tensor = out
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                embed_tensor = out.pooler_output
            else:
                pooled = out.last_hidden_state.mean(dim=1)
                embed_tensor = model.audio_projection(pooled)
            embeddings.append(embed_tensor.squeeze(0).cpu().numpy())
            print(f"  {path.name}")
    return np.stack(embeddings)


def embed_texts(
    texts: list[str],
    model: ClapModel,
    processor: ClapProcessor,
    device: str,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 512), dtype=np.float32)

    print(f"Embedding {len(texts)} prompt texts with CLAP on {device}...")
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        out = model.get_text_features(**inputs)
        if isinstance(out, torch.Tensor):
            embed_tensor = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            embed_tensor = out.pooler_output
        else:
            pooled = out.last_hidden_state.mean(dim=1)
            embed_tensor = model.text_projection(pooled)
    return embed_tensor.cpu().numpy()


def coerce_manifest_row_types(row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    if "comfy_prompt_id" not in coerced and "prompt_slot" not in coerced:
        coerced["comfy_prompt_id"] = coerced.get("prompt_id", "")
        coerced["prompt_id"] = coerced["family_id"]
    coerced.setdefault("prompt_id", coerced["family_id"])
    coerced.setdefault("prompt_label", coerced.get("family_label", coerced["prompt_id"]))
    coerced.setdefault("prompt_slot", 1)
    coerced.setdefault("seed_slot", 1)
    coerced.setdefault("comfy_prompt_id", coerced.get("prompt_id", ""))
    coerced["seed"] = int(coerced["seed"])
    coerced["seed_slot"] = int(coerced["seed_slot"])
    coerced["denoise"] = float(coerced["denoise"])
    coerced["cfg"] = float(coerced["cfg"])
    coerced["steps"] = int(coerced["steps"])
    coerced["family_order"] = int(coerced["family_order"])
    coerced["prompt_slot"] = int(coerced["prompt_slot"])
    seconds = coerced.get("seconds")
    batch_size = coerced.get("batch_size")
    coerced["seconds"] = None if seconds in ("", None) else float(seconds)
    coerced["batch_size"] = None if batch_size in ("", None) else int(batch_size)
    return coerced


def normalized_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [coerce_manifest_row_types(row) for row in manifest_rows if Path(row["local_copy_path"]).exists()]
    rows.sort(key=lambda row: (row["family_order"], row["prompt_slot"], row["seed_slot"], -row["denoise"]))
    return rows


def pairwise_distance_stats(points: np.ndarray) -> dict[str, float]:
    if len(points) < 2:
        return {
            "n": int(len(points)),
            "mean_pairwise_cosine_distance": math.nan,
            "std_pairwise_cosine_distance": math.nan,
            "min_pairwise_cosine_distance": math.nan,
            "max_pairwise_cosine_distance": math.nan,
            "mean_pairwise_cosine_similarity": math.nan,
        }
    dist = cosine_distances(points)
    tri = dist[np.triu_indices(len(points), k=1)]
    return {
        "n": int(len(points)),
        "mean_pairwise_cosine_distance": float(tri.mean()),
        "std_pairwise_cosine_distance": float(tri.std()),
        "min_pairwise_cosine_distance": float(tri.min()),
        "max_pairwise_cosine_distance": float(tri.max()),
        "mean_pairwise_cosine_similarity": float(1.0 - tri.mean()),
    }


def normalized_centroid(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def silhouette_with_null(
    points: np.ndarray,
    labels: np.ndarray,
    n_null: int,
    random_state: int = 0,
) -> dict[str, float]:
    if len(set(labels.tolist())) < 2 or len(points) <= len(set(labels.tolist())):
        return {
            "observed": math.nan,
            "null_mean": math.nan,
            "null_std": math.nan,
            "p_value": math.nan,
            "n_null": 0,
        }

    dist = cosine_distances(points)
    observed = silhouette_score(dist, labels, metric="precomputed")
    rng = np.random.default_rng(random_state)
    null = np.empty(n_null)
    for idx in range(n_null):
        null[idx] = silhouette_score(dist, rng.permutation(labels), metric="precomputed")
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float((null >= observed).mean()),
        "n_null": int(n_null),
    }


def fig_save(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_stem.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


def plot_dispersion(rows: list[dict[str, Any]], out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        grouped[row["family_id"]].append((float(row["denoise"]), float(row["mean_pairwise_cosine_distance"])))
        labels[row["family_id"]] = row["family_label"]

    for idx, family_id in enumerate(sorted(grouped)):
        ordered = sorted(grouped[family_id], reverse=True)
        x = [item[0] for item in ordered]
        y = [item[1] for item in ordered]
        ax.plot(x, y, marker="o", linewidth=1.4, label=labels[family_id])

    ax.set_xlabel("Denoise")
    ax.set_ylabel("Mean pairwise cosine distance")
    ax.set_title("Within-family dispersion by denoise")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Prompt family", loc="best", frameon=True)
    fig.tight_layout()
    fig_save(fig, out_stem)


def plot_distance_to_control(rows: list[dict[str, Any]], control_label: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        grouped[row["family_id"]].append((float(row["denoise"]), float(row["centroid_cosine_distance"])))
        labels[row["family_id"]] = row["family_label"]

    for family_id in sorted(grouped):
        ordered = sorted(grouped[family_id], reverse=True)
        x = [item[0] for item in ordered]
        y = [item[1] for item in ordered]
        ax.plot(x, y, marker="o", linewidth=1.4, label=labels[family_id])

    ax.set_xlabel("Denoise")
    ax.set_ylabel("Centroid cosine distance")
    ax.set_title(f"Distance to {control_label} centroid by denoise")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Prompt family", loc="best", frameon=True)
    fig.tight_layout()
    fig_save(fig, out_stem)


def plot_separation(rows: list[dict[str, Any]], out_stem: Path) -> None:
    ordered = sorted(rows, key=lambda row: -float(row["denoise"]))
    x = [float(row["denoise"]) for row in ordered]
    observed = [float(row["silhouette_observed"]) for row in ordered]
    null_mean = [float(row["silhouette_null_mean"]) for row in ordered]
    null_std = [float(row["silhouette_null_std"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(x, observed, marker="o", linewidth=1.5, color="#1f77b4", label="Observed silhouette")
    ax.plot(x, null_mean, marker="o", linewidth=1.0, linestyle="--", color="#777", label="Shuffled-label null mean")
    ax.fill_between(
        x,
        np.array(null_mean) - np.array(null_std),
        np.array(null_mean) + np.array(null_std),
        color="#bbbbbb",
        alpha=0.25,
        linewidth=0,
        label="Null +/- 1 SD",
    )
    ax.set_xlabel("Denoise")
    ax.set_ylabel("Silhouette (cosine, CLAP space)")
    ax.set_title("Prompt-family separation by denoise")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig_save(fig, out_stem)


def plot_prompt_adherence(rows: list[dict[str, Any]], out_stem: Path) -> None:
    own_rows = [
        row for row in rows
        if row["source_family_id"] == row["target_family_id"]
    ]
    if not own_rows:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in own_rows:
        grouped[row["source_family_id"]].append((float(row["denoise"]), float(row["mean_cosine_similarity"])))
        labels[row["source_family_id"]] = row["source_family_label"]

    for family_id in sorted(grouped):
        ordered = sorted(grouped[family_id], reverse=True)
        x = [item[0] for item in ordered]
        y = [item[1] for item in ordered]
        ax.plot(x, y, marker="o", linewidth=1.4, label=labels[family_id])

    ax.set_xlabel("Denoise")
    ax.set_ylabel("Mean exact-prompt cosine similarity")
    ax.set_title("Exact-prompt CLAP similarity by denoise")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Prompt family", loc="best", frameon=True)
    fig.tight_layout()
    fig_save(fig, out_stem)


def plot_seed_trajectories(
    trajectory_rows: list[dict[str, Any]],
    family_order: list[str],
    family_labels: dict[str, str],
    out_stem: Path,
) -> None:
    families = [family_id for family_id in family_order if family_id in family_labels]
    n_families = len(families)
    if n_families == 0:
        return
    ncols = min(n_families, 3)
    nrows = math.ceil(n_families / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    cmap = plt.get_cmap("viridis")

    denoise_values = sorted({float(row["denoise"]) for row in trajectory_rows})
    denoise_min = min(denoise_values)
    denoise_max = max(denoise_values)
    pc1_values = [float(row["pc1"]) for row in trajectory_rows]
    pc2_values = [float(row["pc2"]) for row in trajectory_rows]
    x_span = max(pc1_values) - min(pc1_values)
    y_span = max(pc2_values) - min(pc2_values)
    x_pad = x_span * 0.06 if x_span else 0.1
    y_pad = y_span * 0.06 if y_span else 0.1
    x_limits = (min(pc1_values) - x_pad, max(pc1_values) + x_pad)
    y_limits = (min(pc2_values) - y_pad, max(pc2_values) + y_pad)

    def color_for_denoise(denoise: float):
        if denoise_max == denoise_min:
            return cmap(0.75)
        norm = (denoise - denoise_min) / (denoise_max - denoise_min)
        return cmap(norm)

    grouped_by_family_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        if row["seed"] == "__centroid__":
            continue
        grouped_by_family_seed[(row["family_id"], int(row["seed"]))].append(row)

    grouped_centroids: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        if row["seed"] == "__centroid__":
            grouped_centroids[row["family_id"]].append(row)

    for idx, family_id in enumerate(families):
        ax = axes[idx // ncols][idx % ncols]
        for (row_family_id, seed), rows_for_seed in grouped_by_family_seed.items():
            if row_family_id != family_id:
                continue
            ordered = sorted(rows_for_seed, key=lambda row: -float(row["denoise"]))
            xs = [float(row["pc1"]) for row in ordered]
            ys = [float(row["pc2"]) for row in ordered]
            ax.plot(xs, ys, color="#999999", alpha=0.45, linewidth=0.8, zorder=1)
            for row in ordered:
                ax.scatter(
                    float(row["pc1"]),
                    float(row["pc2"]),
                    s=28,
                    color=color_for_denoise(float(row["denoise"])),
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=2,
                )

        centroid_rows = sorted(grouped_centroids.get(family_id, []), key=lambda row: -float(row["denoise"]))
        if centroid_rows:
            cx = [float(row["pc1"]) for row in centroid_rows]
            cy = [float(row["pc2"]) for row in centroid_rows]
            ax.plot(cx, cy, color="#222222", linewidth=1.8, marker="s", markersize=4, label="Centroid", zorder=3)

        ax.set_title(family_labels[family_id])
        ax.set_xlabel("PCA dim 1")
        if idx % ncols == 0:
            ax.set_ylabel("PCA dim 2")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for extra_idx in range(n_families, nrows * ncols):
        axes[extra_idx // ncols][extra_idx % ncols].axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color_for_denoise(denoise),
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=6,
            label=f"denoise={denoise:g}",
        )
        for denoise in sorted(denoise_values, reverse=True)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig_save(fig, out_stem)


def analyze(config: dict[str, Any], experiment_root: Path, manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = normalized_rows(manifest_rows)
    if not rows:
        raise SystemExit("No local audio files found for analysis.")

    family_labels: dict[str, str] = {}
    family_order_map: dict[str, int] = {}
    family_order: list[str] = []
    prompt_labels: dict[str, str] = {}
    prompt_family_map: dict[str, str] = {}
    prompt_text_map: dict[str, str] = {}
    prompt_order: list[str] = []
    for row in rows:
        family_id = row["family_id"]
        family_labels[family_id] = row["family_label"]
        family_order_map[family_id] = int(row["family_order"])
        if family_id not in family_order:
            family_order.append(family_id)
        prompt_id = row["prompt_id"]
        prompt_labels[prompt_id] = row["prompt_label"]
        prompt_family_map[prompt_id] = family_id
        prompt_text_map[prompt_id] = row["prompt_text"]
        if prompt_id not in prompt_order:
            prompt_order.append(prompt_id)

    (experiment_root / "embeddings").mkdir(parents=True, exist_ok=True)
    (experiment_root / "metrics").mkdir(parents=True, exist_ok=True)
    (experiment_root / "figures").mkdir(parents=True, exist_ok=True)

    nonempty_prompts: list[tuple[str, str, str, str, str]] = []
    for prompt_id in prompt_order:
        prompt_text = prompt_text_map[prompt_id].strip()
        if prompt_text:
            family_id = prompt_family_map[prompt_id]
            nonempty_prompts.append((
                prompt_id,
                prompt_labels[prompt_id],
                family_id,
                family_labels[family_id],
                prompt_text,
            ))

    target_prompt_ids = [item[0] for item in nonempty_prompts]
    target_prompt_labels = [item[1] for item in nonempty_prompts]
    target_prompt_family_ids = [item[2] for item in nonempty_prompts]
    target_prompt_family_labels = [item[3] for item in nonempty_prompts]
    prompt_texts = [item[4] for item in nonempty_prompts]

    audio_paths = [Path(row["local_copy_path"]) for row in rows]
    embedding_cache_path = experiment_root / "embeddings" / "clap_embeddings.npz"
    cached_embeddings_loaded = False
    if embedding_cache_path.exists():
        with np.load(embedding_cache_path) as cached:
            cached_filenames = [str(value) for value in cached["filenames"].tolist()]
            cached_prompt_texts = [str(value) for value in cached["prompt_texts"].tolist()]
            if cached_filenames == [path.name for path in audio_paths] and cached_prompt_texts == prompt_texts:
                embeddings = cached["embeddings"]
                embeddings_norm = cached["embeddings_norm"]
                text_embeddings = cached["text_embeddings"]
                text_embeddings_norm = cached["text_embeddings_norm"]
                audio_text_similarity = cached["audio_text_similarity"]
                cached_embeddings_loaded = True
                print(f"Loaded cached CLAP embeddings from {embedding_cache_path}")

    if not cached_embeddings_loaded:
        model, processor, device = load_clap()
        embeddings = embed_audio_files(audio_paths, model, processor, device)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        text_embeddings = embed_texts(prompt_texts, model, processor, device)
        if len(text_embeddings) > 0:
            text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            audio_text_similarity = embeddings_norm @ text_embeddings_norm.T
        else:
            text_embeddings_norm = np.empty((0, embeddings_norm.shape[1]), dtype=np.float32)
            audio_text_similarity = np.empty((len(rows), 0), dtype=np.float32)
    elif len(text_embeddings) == 0:
        text_embeddings_norm = np.empty((0, embeddings_norm.shape[1]), dtype=np.float32)
        audio_text_similarity = np.empty((len(rows), 0), dtype=np.float32)

    if not cached_embeddings_loaded:
        np.savez(
            embedding_cache_path,
            embeddings=embeddings,
            embeddings_norm=embeddings_norm,
            filenames=np.array([path.name for path in audio_paths]),
            family_ids=np.array([row["family_id"] for row in rows]),
            family_labels=np.array([row["family_label"] for row in rows]),
            prompt_ids=np.array([row["prompt_id"] for row in rows]),
            prompt_labels=np.array([row["prompt_label"] for row in rows]),
            prompt_slots=np.array([int(row["prompt_slot"]) for row in rows]),
            denoise=np.array([float(row["denoise"]) for row in rows]),
            seeds=np.array([int(row["seed"]) for row in rows]),
            seed_slots=np.array([int(row["seed_slot"]) for row in rows]),
            target_prompt_ids=np.array(target_prompt_ids),
            target_prompt_labels=np.array(target_prompt_labels),
            target_prompt_family_ids=np.array(target_prompt_family_ids),
            target_prompt_family_labels=np.array(target_prompt_family_labels),
            prompt_texts=np.array(prompt_texts),
            text_embeddings=text_embeddings,
            text_embeddings_norm=text_embeddings_norm,
            audio_text_similarity=audio_text_similarity,
        )

    actual_rows: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["local_copy_path"])
        actual = extract_actual_metadata(path)
        match_ok, mismatch_summary = compare_expected_actual(JobSpec(
            experiment_name=row["experiment_name"],
            family_id=row["family_id"],
            family_label=row["family_label"],
            prompt_id=row["prompt_id"],
            prompt_label=row["prompt_label"],
            prompt_text=row["prompt_text"],
            prompt_slot=int(row["prompt_slot"]),
            negative_prompt=row["negative_prompt"],
            seed=int(row["seed"]),
            seed_slot=int(row["seed_slot"]),
            denoise=float(row["denoise"]),
            cfg=float(row["cfg"]),
            steps=int(row["steps"]),
            sampler_name=row["sampler_name"],
            scheduler=row["scheduler"],
            seconds=float(row["seconds"]) if row["seconds"] not in ("", None) else None,
            batch_size=int(row["batch_size"]) if row["batch_size"] not in ("", None) else None,
            comfy_prefix=row["comfy_prefix"],
            local_filename=row["local_filename"],
            comfy_prompt_id=row["comfy_prompt_id"],
            family_order=int(row["family_order"]),
        ), actual)
        actual_rows.append({
            "local_filename": path.name,
            "family_id": row["family_id"],
            "family_label": row["family_label"],
            "prompt_id": row["prompt_id"],
            "prompt_label": row["prompt_label"],
            "prompt_slot": row["prompt_slot"],
            "seed": row["seed"],
            "seed_slot": row["seed_slot"],
            "denoise": row["denoise"],
            "intended_prompt": row["prompt_text"],
            "actual_prompt": actual["actual_prompt"],
            "intended_cfg": row["cfg"],
            "actual_cfg": actual["actual_cfg"],
            "intended_steps": row["steps"],
            "actual_steps": actual["actual_steps"],
            "intended_denoise": row["denoise"],
            "actual_denoise": actual["actual_denoise"],
            "intended_sampler_name": row["sampler_name"],
            "actual_sampler_name": actual["actual_sampler_name"],
            "intended_scheduler": row["scheduler"],
            "actual_scheduler": actual["actual_scheduler"],
            "intended_seed": row["seed"],
            "actual_seed": actual["actual_seed"],
            "metadata_ok": actual["metadata_ok"],
            "matches_intended": match_ok,
            "mismatch_summary": mismatch_summary,
        })

    within_family_rows: list[dict[str, Any]] = []
    within_prompt_rows: list[dict[str, Any]] = []
    rows_by_family_denoise: dict[tuple[str, float], list[int]] = defaultdict(list)
    rows_by_prompt_denoise: dict[tuple[str, float], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        family_id = row["family_id"]
        rows_by_family_denoise[(family_id, float(row["denoise"]))].append(idx)
        rows_by_prompt_denoise[(row["prompt_id"], float(row["denoise"]))].append(idx)

    audio_text_similarity_rows: list[dict[str, Any]] = []
    prompt_adherence_rows: list[dict[str, Any]] = []
    prompt_text_adherence_rows: list[dict[str, Any]] = []
    prompt_confusion_rows: list[dict[str, Any]] = []

    if target_prompt_ids:
        for idx, row in enumerate(rows):
            per_file = {
                "local_filename": Path(row["local_copy_path"]).name,
                "family_id": row["family_id"],
                "family_label": row["family_label"],
                "prompt_id": row["prompt_id"],
                "prompt_label": row["prompt_label"],
                "prompt_slot": row["prompt_slot"],
                "seed": row["seed"],
                "seed_slot": row["seed_slot"],
                "denoise": row["denoise"],
            }
            similarities = audio_text_similarity[idx]
            best_idx = int(np.argmax(similarities))
            per_file["best_matching_prompt_id"] = target_prompt_ids[best_idx]
            per_file["best_matching_prompt_label"] = target_prompt_labels[best_idx]
            per_file["best_matching_prompt_family_id"] = target_prompt_family_ids[best_idx]
            per_file["best_matching_prompt_family_label"] = target_prompt_family_labels[best_idx]
            per_file["best_matching_prompt_cosine_similarity"] = float(similarities[best_idx])
            for prompt_idx, target_prompt_id in enumerate(target_prompt_ids):
                per_file[f"similarity_to_{target_prompt_id}"] = float(similarities[prompt_idx])
            audio_text_similarity_rows.append(per_file)

        denoise_values = sorted({float(row["denoise"]) for row in rows}, reverse=True)
        for denoise in denoise_values:
            denoise_idxs = [idx for idx, row in enumerate(rows) if float(row["denoise"]) == denoise]
            for family_id in family_order:
                family_idxs = [idx for idx in denoise_idxs if rows[idx]["family_id"] == family_id]
                if not family_idxs:
                    continue
                source_prompt_ids = sorted({rows[idx]["prompt_id"] for idx in family_idxs})
                exact_prompt_sims: list[float] = []
                for source_prompt_id in source_prompt_ids:
                    prompt_idxs = [idx for idx in family_idxs if rows[idx]["prompt_id"] == source_prompt_id]
                    if source_prompt_id in target_prompt_ids:
                        target_idx = target_prompt_ids.index(source_prompt_id)
                        exact_prompt_sims.extend(float(value) for value in audio_text_similarity[prompt_idxs, target_idx])
                    for target_idx, target_prompt_id in enumerate(target_prompt_ids):
                        sims = audio_text_similarity[prompt_idxs, target_idx]
                        prompt_text_adherence_rows.append({
                            "source_family_id": family_id,
                            "source_family_label": family_labels[family_id],
                            "source_prompt_id": source_prompt_id,
                            "source_prompt_label": prompt_labels[source_prompt_id],
                            "target_prompt_id": target_prompt_id,
                            "target_prompt_label": target_prompt_labels[target_idx],
                            "target_prompt_family_id": target_prompt_family_ids[target_idx],
                            "target_prompt_family_label": target_prompt_family_labels[target_idx],
                            "denoise": denoise,
                            "n": len(prompt_idxs),
                            "mean_cosine_similarity": float(sims.mean()),
                            "std_cosine_similarity": float(sims.std()),
                        })
                if exact_prompt_sims:
                    sims_array = np.array(exact_prompt_sims, dtype=np.float32)
                    prompt_adherence_rows.append({
                        "source_family_id": family_id,
                        "source_family_label": family_labels[family_id],
                        "target_family_id": family_id,
                        "target_family_label": family_labels[family_id],
                        "denoise": denoise,
                        "n": int(len(sims_array)),
                        "mean_cosine_similarity": float(sims_array.mean()),
                        "std_cosine_similarity": float(sims_array.std()),
                    })

                best_matches = [
                    target_prompt_family_ids[int(np.argmax(audio_text_similarity[idx]))]
                    for idx in family_idxs
                ]
                for target_family_id in family_order:
                    count = sum(match == target_family_id for match in best_matches)
                    prompt_confusion_rows.append({
                        "source_family_id": family_id,
                        "source_family_label": family_labels[family_id],
                        "best_matching_prompt_family_id": target_family_id,
                        "best_matching_prompt_family_label": family_labels[target_family_id],
                        "denoise": denoise,
                        "count": count,
                        "proportion": float(count / len(best_matches)),
                    })

    for (family_id, denoise), idxs in sorted(
        rows_by_family_denoise.items(),
        key=lambda item: (family_order_map[item[0][0]], -item[0][1]),
    ):
        stats = pairwise_distance_stats(embeddings_norm[idxs])
        within_family_rows.append({
            "family_id": family_id,
            "family_label": family_labels[family_id],
            "denoise": denoise,
            **stats,
        })

    for (prompt_id, denoise), idxs in sorted(
        rows_by_prompt_denoise.items(),
        key=lambda item: (
            family_order_map[prompt_family_map[item[0][0]]],
            min(int(rows[idx]["prompt_slot"]) for idx in item[1]),
            -item[0][1],
        ),
    ):
        stats = pairwise_distance_stats(embeddings_norm[idxs])
        family_id = prompt_family_map[prompt_id]
        within_prompt_rows.append({
            "family_id": family_id,
            "family_label": family_labels[family_id],
            "prompt_id": prompt_id,
            "prompt_label": prompt_labels[prompt_id],
            "prompt_text": prompt_text_map[prompt_id],
            "denoise": denoise,
            **stats,
        })

    control_family_id = slugify(config["control_family_id"])
    distance_to_control_rows: list[dict[str, Any]] = []
    for denoise in sorted({float(row["denoise"]) for row in rows}, reverse=True):
        control_idxs = rows_by_family_denoise.get((control_family_id, denoise), [])
        if not control_idxs:
            continue
        control_centroid = normalized_centroid(embeddings_norm[control_idxs])
        for family_id in family_order:
            if family_id == control_family_id:
                continue
            idxs = rows_by_family_denoise.get((family_id, denoise), [])
            if not idxs:
                continue
            family_points = embeddings_norm[idxs]
            family_centroid = normalized_centroid(family_points)
            sample_dist = cosine_distances(family_points, control_centroid.reshape(1, -1)).reshape(-1)
            distance_to_control_rows.append({
                "family_id": family_id,
                "family_label": family_labels[family_id],
                "control_family_id": control_family_id,
                "control_family_label": family_labels.get(control_family_id, control_family_id),
                "denoise": denoise,
                "n": len(idxs),
                "centroid_cosine_distance": float(1.0 - float(family_centroid @ control_centroid)),
                "mean_sample_to_control_distance": float(sample_dist.mean()),
                "std_sample_to_control_distance": float(sample_dist.std()),
            })

    separation_rows: list[dict[str, Any]] = []
    for denoise in sorted({float(row["denoise"]) for row in rows}, reverse=True):
        idxs = [idx for idx, row in enumerate(rows) if float(row["denoise"]) == denoise]
        points = embeddings_norm[idxs]
        labels = np.array([rows[idx]["family_id"] for idx in idxs])
        sil = silhouette_with_null(points, labels, n_null=int(config["null_samples"]))

        centroid_distance_matrix: dict[str, dict[str, float]] = {}
        for family_id_a in family_order:
            centroid_distance_matrix[family_id_a] = {}
            idxs_a = [idx for idx in idxs if rows[idx]["family_id"] == family_id_a]
            if not idxs_a:
                continue
            centroid_a = normalized_centroid(embeddings_norm[idxs_a])
            for family_id_b in family_order:
                idxs_b = [idx for idx in idxs if rows[idx]["family_id"] == family_id_b]
                if not idxs_b:
                    continue
                centroid_b = normalized_centroid(embeddings_norm[idxs_b])
                centroid_distance_matrix[family_id_a][family_id_b] = float(1.0 - float(centroid_a @ centroid_b))

        separation_rows.append({
            "denoise": denoise,
            "n_clips": len(idxs),
            "n_families": len(set(labels.tolist())),
            "silhouette_observed": sil["observed"],
            "silhouette_null_mean": sil["null_mean"],
            "silhouette_null_std": sil["null_std"],
            "silhouette_p_value": sil["p_value"],
            "null_samples": sil["n_null"],
            "centroid_distance_matrix_json": json.dumps(centroid_distance_matrix, sort_keys=True),
        })

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_norm)
    trajectory_rows: list[dict[str, Any]] = []
    for row, (pc1, pc2) in zip(rows, coords):
        trajectory_rows.append({
            "family_id": row["family_id"],
            "family_label": row["family_label"],
            "prompt_id": row["prompt_id"],
            "prompt_label": row["prompt_label"],
            "prompt_slot": row["prompt_slot"],
            "seed": row["seed"],
            "seed_slot": row["seed_slot"],
            "denoise": row["denoise"],
            "local_filename": row["local_filename"],
            "pc1": float(pc1),
            "pc2": float(pc2),
        })
    for family_id in family_order:
        for denoise in sorted({float(row["denoise"]) for row in rows}, reverse=True):
            idxs = rows_by_family_denoise.get((family_id, denoise), [])
            if not idxs:
                continue
            centroid = coords[idxs].mean(axis=0)
            trajectory_rows.append({
                "family_id": family_id,
                "family_label": family_labels[family_id],
                "prompt_id": "__family_centroid__",
                "prompt_label": "Family centroid",
                "prompt_slot": "",
                "seed": "__centroid__",
                "seed_slot": "",
                "denoise": denoise,
                "local_filename": "",
                "pc1": float(centroid[0]),
                "pc2": float(centroid[1]),
            })

    figures_dir = experiment_root / "figures"
    plot_dispersion(within_family_rows, figures_dir / "dispersion_by_denoise")
    plot_distance_to_control(
        distance_to_control_rows,
        control_label=family_labels.get(control_family_id, control_family_id),
        out_stem=figures_dir / "distance_to_unconditioned_by_denoise",
    )
    plot_separation(separation_rows, figures_dir / "separation_by_denoise")
    plot_prompt_adherence(prompt_adherence_rows, figures_dir / "prompt_adherence_by_denoise")
    plot_seed_trajectories(
        trajectory_rows,
        family_order=family_order,
        family_labels=family_labels,
        out_stem=figures_dir / "seed_matched_trajectories",
    )

    summary = {
        "experiment_name": config["experiment_name"],
        "created_at_utc": utc_now_iso(),
        "control_family_id": control_family_id,
        "n_jobs": len(rows),
        "n_prompt_families": len(family_order),
        "n_prompt_entries": len(prompt_order),
        "n_nonempty_prompt_texts": len(nonempty_prompts),
        "n_prompt_texts": len(nonempty_prompts),
        "family_order": family_order,
        "prompt_order": prompt_order,
        "denoise_values": sorted({float(row["denoise"]) for row in rows}, reverse=True),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_[:2]],
        "files": {
            "manifest_csv": str(experiment_root / "manifest.csv"),
            "actual_metadata_csv": str(experiment_root / "actual_metadata.csv"),
            "within_family_dispersion_csv": str(experiment_root / "metrics" / "within_family_dispersion.csv"),
            "within_prompt_dispersion_csv": str(experiment_root / "metrics" / "within_prompt_dispersion.csv"),
            "distance_to_control_csv": str(experiment_root / "metrics" / "distance_to_unconditioned.csv"),
            "separation_csv": str(experiment_root / "metrics" / "separation_by_denoise.csv"),
            "trajectory_csv": str(experiment_root / "metrics" / "trajectory_coords.csv"),
            "audio_text_similarity_csv": str(experiment_root / "metrics" / "audio_text_similarity.csv"),
            "prompt_adherence_csv": str(experiment_root / "metrics" / "prompt_adherence_by_denoise.csv"),
            "prompt_text_adherence_csv": str(experiment_root / "metrics" / "prompt_text_adherence_by_denoise.csv"),
            "prompt_confusion_csv": str(experiment_root / "metrics" / "prompt_confusion_by_denoise.csv"),
        },
        "within_family_dispersion": within_family_rows,
        "within_prompt_dispersion": within_prompt_rows,
        "distance_to_control": distance_to_control_rows,
        "separation_by_denoise": [
            {
                key: value
                for key, value in row.items()
                if key != "centroid_distance_matrix_json"
            } | {
                "centroid_distance_matrix": json.loads(row["centroid_distance_matrix_json"])
            }
            for row in separation_rows
        ],
        "prompt_texts": [
            {
                "prompt_id": prompt_id,
                "prompt_label": prompt_label,
                "family_id": family_id,
                "family_label": family_label,
                "prompt_text": prompt_text,
            }
            for prompt_id, prompt_label, family_id, family_label, prompt_text in nonempty_prompts
        ],
        "prompt_adherence_by_denoise": prompt_adherence_rows,
        "prompt_text_adherence_by_denoise": prompt_text_adherence_rows,
        "prompt_confusion_by_denoise": prompt_confusion_rows,
    }

    write_csv(experiment_root / "actual_metadata.csv", actual_rows)
    write_csv(experiment_root / "metrics" / "within_family_dispersion.csv", within_family_rows)
    write_csv(experiment_root / "metrics" / "within_prompt_dispersion.csv", within_prompt_rows)
    write_csv(experiment_root / "metrics" / "distance_to_unconditioned.csv", distance_to_control_rows)
    write_csv(experiment_root / "metrics" / "separation_by_denoise.csv", separation_rows)
    write_csv(experiment_root / "metrics" / "trajectory_coords.csv", trajectory_rows)
    write_csv(experiment_root / "metrics" / "audio_text_similarity.csv", audio_text_similarity_rows)
    write_csv(experiment_root / "metrics" / "prompt_adherence_by_denoise.csv", prompt_adherence_rows)
    write_csv(experiment_root / "metrics" / "prompt_text_adherence_by_denoise.csv", prompt_text_adherence_rows)
    write_csv(experiment_root / "metrics" / "prompt_confusion_by_denoise.csv", prompt_confusion_rows)
    write_json(experiment_root / "summary.json", summary)
    write_paper_summary(experiment_root / "paper_summary.md", config, summary, actual_rows)
    return summary


def write_paper_summary(
    path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    actual_rows: list[dict[str, Any]],
) -> None:
    separation = summary["separation_by_denoise"]
    strongest_sep = max(
        (row for row in separation if not math.isnan(row["silhouette_observed"])),
        key=lambda row: row["silhouette_observed"],
        default=None,
    )
    weakest_sep = min(
        (row for row in separation if not math.isnan(row["silhouette_observed"])),
        key=lambda row: row["silhouette_observed"],
        default=None,
    )

    distance_rows = summary["distance_to_control"]
    prompt_adherence_rows = summary.get("prompt_adherence_by_denoise", [])
    prompt_confusion_rows = summary.get("prompt_confusion_by_denoise", [])
    control_family_id = summary["control_family_id"]
    nonmatching = [row for row in actual_rows if not row["matches_intended"]]

    lines = [
        f"# {summary['experiment_name']} audit summary",
        "",
        f"- Rendered/analyzed {summary['n_jobs']} clips across {summary['n_prompt_families']} prompt families, "
        f"{summary.get('n_prompt_entries', summary.get('n_prompt_texts', 0))} prompt entries "
        f"({summary.get('n_nonempty_prompt_texts', summary.get('n_prompt_texts', 0))} non-empty text prompts), "
        f"and {len(summary['denoise_values'])} denoise values.",
        f"- Control family: `{control_family_id}`.",
    ]
    if strongest_sep is not None:
        lines.append(
            "- Strongest prompt-family separation occurred at "
            f"denoise `{strongest_sep['denoise']:g}` "
            f"(silhouette `{strongest_sep['silhouette_observed']:.3f}`, "
            f"`p={strongest_sep['silhouette_p_value']:.3f}`)."
        )
    if weakest_sep is not None:
        lines.append(
            "- Weakest prompt-family separation occurred at "
            f"denoise `{weakest_sep['denoise']:g}` "
            f"(silhouette `{weakest_sep['silhouette_observed']:.3f}`, "
            f"`p={weakest_sep['silhouette_p_value']:.3f}`)."
        )

    for family in config["prompt_families"]:
        family_id = slugify(family["id"])
        family_label = family["label"]
        family_dist = [row for row in distance_rows if row["family_id"] == family_id]
        if not family_dist:
            continue
        full = max(family_dist, key=lambda row: row["denoise"])
        partial = min(family_dist, key=lambda row: row["denoise"])
        lines.append(
            f"- `{family_label}` centroid distance to `{control_family_id}` moved from "
            f"`{full['centroid_cosine_distance']:.3f}` at denoise `{full['denoise']:g}` to "
            f"`{partial['centroid_cosine_distance']:.3f}` at denoise `{partial['denoise']:g}`."
        )

    nonempty_families = []
    for family in config["prompt_families"]:
        if any(prompt["prompt"].strip() for prompt in family["prompts"]):
            nonempty_families.append(family)
    for family in nonempty_families:
        family_id = slugify(family["id"])
        family_label = family["label"]
        own_rows = [
            row for row in prompt_adherence_rows
            if row["source_family_id"] == family_id
            and row["target_family_id"] == family_id
        ]
        if own_rows:
            full = max(own_rows, key=lambda row: row["denoise"])
            partial = min(own_rows, key=lambda row: row["denoise"])
            lines.append(
                f"- `{family_label}` exact-prompt CLAP similarity moved from "
                f"`{full['mean_cosine_similarity']:.3f}` at denoise `{full['denoise']:g}` to "
                f"`{partial['mean_cosine_similarity']:.3f}` at denoise `{partial['denoise']:g}`."
            )

        win_rows = [
            row for row in prompt_confusion_rows
            if row["source_family_id"] == family_id
            and row["best_matching_prompt_family_id"] == family_id
        ]
        if win_rows:
            full = max(win_rows, key=lambda row: row["denoise"])
            partial = min(win_rows, key=lambda row: row["denoise"])
            lines.append(
                f"- `{family_label}` best-match win rate against the prompt-text set moved from "
                f"`{full['proportion']:.3f}` at denoise `{full['denoise']:g}` to "
                f"`{partial['proportion']:.3f}` at denoise `{partial['denoise']:g}`."
            )

    control_prompt_rows = [
        row for row in prompt_adherence_rows
        if row["source_family_id"] == control_family_id
    ]
    if control_prompt_rows:
        high_denoise = max(control_prompt_rows, key=lambda row: row["denoise"])["denoise"]
        low_denoise = min(control_prompt_rows, key=lambda row: row["denoise"])["denoise"]
        for denoise in [high_denoise, low_denoise]:
            rows_at_denoise = [row for row in control_prompt_rows if row["denoise"] == denoise]
            best = max(rows_at_denoise, key=lambda row: row["mean_cosine_similarity"])
            lines.append(
                f"- At denoise `{denoise:g}`, the `{control_family_id}` family is closest in text space to "
                f"`{best['target_family_label']}` "
                f"(mean cosine similarity `{best['mean_cosine_similarity']:.3f}`)."
            )

    if nonmatching:
        lines.append(f"- {len(nonmatching)} files showed intended-vs-embedded metadata mismatches; inspect `actual_metadata.csv` before writing claims about fixed conditions.")
    else:
        lines.append("- All analyzed files matched the intended sampler/prompt settings in embedded FLAC metadata.")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build the manifest but do not queue or analyze.")
    parser.add_argument("--queue-only", action="store_true", help="Queue/harvest renders but skip CLAP analysis.")
    parser.add_argument("--analysis-only", action="store_true", help="Analyze existing local audit audio only.")
    parser.add_argument("--rerun-existing", action="store_true", help="Requeue jobs even if local copies already exist.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval while waiting for ComfyUI renders.")
    parser.add_argument("--timeout-minutes", type=float, default=180.0, help="Maximum time to wait for queued jobs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    experiment_root = audit_root(config)
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json(experiment_root / "config_snapshot.json", config)

    jobs = build_jobs(config)
    manifest_rows = [asdict(job) | {"local_copy_path": str(local_audio_path(experiment_root, job))} for job in jobs]
    write_csv(experiment_root / "manifest.csv", manifest_rows)
    print(f"Prepared {len(jobs)} jobs in {experiment_root}")

    if args.dry_run:
        print("Dry run only; manifest written, no rendering or analysis performed.")
        return

    if args.analysis_only and args.queue_only:
        raise SystemExit("Choose either --analysis-only or --queue-only, not both.")

    if not args.analysis_only:
        workflow_template = load_workflow(Path(config["workflow_path"]))
        manifest_rows = gather_existing_or_queue(
            config=config,
            jobs=jobs,
            workflow_template=workflow_template,
            experiment_root=experiment_root,
            rerun_existing=args.rerun_existing,
            poll_seconds=args.poll_seconds,
            timeout_minutes=args.timeout_minutes,
        )
        write_csv(experiment_root / "manifest.csv", manifest_rows)
    else:
        existing_manifest = experiment_root / "manifest.csv"
        if existing_manifest.exists():
            with existing_manifest.open() as f:
                manifest_rows = list(csv.DictReader(f))
        else:
            manifest_rows = [asdict(job) | {"local_copy_path": str(local_audio_path(experiment_root, job))} for job in jobs]

    if args.queue_only:
        print("Queue/harvest complete; skipping CLAP analysis because --queue-only was set.")
        return

    summary = analyze(config, experiment_root, manifest_rows)
    print(
        f"\nFinished audit '{summary['experiment_name']}'. "
        f"Summary: {experiment_root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
