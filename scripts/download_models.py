"""Download the M0 model files into ComfyUI/models (run on Colab).

  python scripts/download_models.py --comfy-dir /content/ComfyUI --precision auto

Needs HF_TOKEN with access granted to the gated Lightricks repos ("Agree and access" once in a
browser). `--precision auto` picks bf16 on >= 70 GB GPUs, nvfp4 (4-bit transformer) on Blackwell GPUs under 40 GB, and
Comfy int8 otherwise; the text encoder is int8 whenever the transformer is not bf16. Writes the
chosen filenames to <comfy-dir>/timeflight_models.json; app/run_m0.py reads it.
All files are logged in assets.jsonl (see docs/decisions.md for license notes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

LTX_REPO = "Lightricks/LTX-2.5"
UNION_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
UNION_FILE = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

BF16 = {
    "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
}
COMMON = {
    "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}
UPSCALER = "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"


def gpu_memory_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return float(out.splitlines()[0]) / 1024


def pick_int8(files: list[str], folder: str, must: list[str]) -> str:
    """Comfy int8 filenames are not documented; find them by pattern and fail loudly."""
    hits = [f for f in files if f.startswith(folder + "/") and "int8" in f.lower()
            and all(m in f.lower() for m in must)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one int8 file in {folder} matching {must}, found {hits}. "
                         f"Files in {folder}: {[f for f in files if f.startswith(folder + '/')]}")
    return hits[0]


def gpu_compute_capability() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return float(out.splitlines()[0])


def pick_quant(files: list[str], folder: str, must: list[str], tag: str) -> str:
    """Quantised filenames are not documented; find them by pattern (e.g. tag 'nvfp4') and fail loudly."""
    hits = [f for f in files if f.startswith(folder + "/") and tag in f.lower() and all(m in f.lower() for m in must)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one {tag} file in {folder} matching {must}, found {hits}")
    return hits[0]


def resolve_precision(precision: str) -> str:
    if precision != "auto":
        return precision
    mem = gpu_memory_gb()
    if mem >= 70:
        return "bf16"
    if mem < 40 and gpu_compute_capability() >= 10.0:  # FP4 tensor cores: Blackwell (sm_100 / sm_120)
        return "nvfp4"
    return "int8"


def choose(files: list[str], precision: str) -> dict[str, str]:
    """Model files for a resolved precision (bf16 | int8 | nvfp4), without the common VAEs."""
    if precision == "bf16":
        return dict(BF16)
    transformer = (pick_quant(files, "diffusion_models", ["distilled"], "nvfp4") if precision == "nvfp4"
                   else pick_int8(files, "diffusion_models", ["distilled"]))
    return {"transformer": transformer, "text_encoder": pick_int8(files, "text_encoders", ["gemma"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-dir", type=Path, required=True)
    p.add_argument("--precision", choices=["auto", "bf16", "int8", "nvfp4"], default="auto")
    p.add_argument("--with-upscaler", action="store_true", help="stage 2 (not needed for M0)")
    p.add_argument("--list", action="store_true", help="only list repo files")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    files = HfApi().list_repo_files(LTX_REPO, token=token)
    if args.list:
        print("\n".join(files))
        return

    precision = resolve_precision(args.precision)
    chosen = choose(files, precision)
    chosen.update(COMMON)
    if args.with_upscaler:
        chosen["upscaler"] = UPSCALER

    models = args.comfy_dir / "models"
    for key, repo_path in chosen.items():
        if repo_path not in files:
            raise SystemExit(f"{repo_path} not found in {LTX_REPO}; run with --list")
        print(f"[{key}] {repo_path}", flush=True)
        hf_hub_download(LTX_REPO, repo_path, local_dir=models, token=token)

    print(f"[union_lora] {UNION_FILE}", flush=True)
    hf_hub_download(UNION_REPO, UNION_FILE, local_dir=models / "loras", token=token)

    names = {k: Path(v).name for k, v in chosen.items()}
    names["union_lora"] = UNION_FILE
    names["precision"] = precision
    (args.comfy_dir / "timeflight_models.json").write_text(json.dumps(names, indent=2))
    print(json.dumps(names, indent=2))


if __name__ == "__main__":
    main()
