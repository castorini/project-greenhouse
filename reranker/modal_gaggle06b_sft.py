"""Reproduce the released 0.6B Gaggle RLHN-250K SFT on one Modal H200.

Run from this repository checkout with a Modal profile:

    modal run reranker/modal_gaggle06b_sft.py --stage stage
    modal run reranker/modal_gaggle06b_sft.py --stage preflight  # first attempt only
    modal deploy reranker/modal_gaggle06b_sft.py
    python -c 'import modal; c = modal.Function.from_name("greenhouse-gaggle06b-rlhn-sft", "train").spawn(); print(c.object_id)'

The staging function checks the released backbone and prepared RLHN-250K data
before training. Training output goes to a separate Volume. No automatic retries
are configured.
The 13-hour process limit plus the 14-hour Modal timeout cap a resumed attempt
below the remaining $100 budget at the verified $5.1744/hour resource rate.
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO_COMMIT = "9c878928b89245d411f349e89def7100d18f3611"
BASE_REVISION = "c3c0d9ff865bcaf23b268b6d88b3aa3d05ef9768"
BASE_SHA256 = "4e45e1458d00d8d2b869091404ee9c75966431425b972af3364bcd5a5da3ae9d"
DATA_SHA256 = "c3743e83920a54d260483573d9d3176e165dcc3a34e398413ecc49f75b9c06d0"
EXPECTED_STEPS = 1928
MAX_PROCESS_SECONDS = 13 * 3600
RUN_NAME = "gaggle06b_lce_rlhn250k_20261125"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git init /opt/project-greenhouse",
        "git -C /opt/project-greenhouse remote add origin https://github.com/castorini/project-greenhouse.git",
        f"git -C /opt/project-greenhouse fetch --depth 1 origin {REPO_COMMIT}",
        "git -C /opt/project-greenhouse checkout --detach FETCH_HEAD",
        "python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130",
        "python -m pip install -r /opt/project-greenhouse/reranker/requirements.txt",
        "bash /opt/project-greenhouse/pretrain/continue/setup.sh",
    )
    .env({"HF_HOME": "/probe/hf", "WANDB_MODE": "offline"})
)
app = modal.App("greenhouse-gaggle06b-rlhn-sft", image=image)
inputs = modal.Volume.from_name("greenhouse-sft-bounded-probe", create_if_missing=True)
outputs = modal.Volume.from_name("greenhouse-gaggle06b-sft", create_if_missing=True)

reranker_dir = Path("/opt/project-greenhouse/reranker")
model_dir = reranker_dir / "models/gaggle-nanochat-0.6b-pretrained-climbmix-20261121"
model_source = Path("/probe/models/gaggle-nanochat-0.6b-pretrained-climbmix-20261121")
data_file = reranker_dir / "data/rlhn250k_prepared_climbmix_lens/train.parquet"
run_dir = Path("/results") / RUN_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def link_inputs() -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not model_dir.exists():
        model_dir.symlink_to(model_source, target_is_directory=True)
    data_link = reranker_dir / "data"
    if not data_link.exists():
        data_link.symlink_to("/probe/data", target_is_directory=True)


def check_inputs() -> dict:
    link_inputs()
    if sha256(model_dir / "model_229067.pt") != BASE_SHA256:
        raise RuntimeError("Released 0.6B backbone SHA256 mismatch")
    if sha256(data_file) != DATA_SHA256:
        raise RuntimeError("Prepared RLHN-250K SHA256 mismatch")
    meta = json.loads((model_dir / "meta_229067.json").read_text())
    arch = meta["model_config"]
    if (meta["step"], arch["n_layer"], arch["n_embd"], arch["ve_every"]) != (229067, 28, 1024, 7):
        raise RuntimeError("Backbone metadata does not describe the released 0.6B architecture")
    from pyarrow import parquet

    rows = parquet.ParquetFile(data_file).metadata.num_rows
    if rows != 247534:
        raise RuntimeError(f"Expected 247,534 prepared RLHN rows; found {rows}")
    return {"repo_commit": REPO_COMMIT, "base_revision": BASE_REVISION,
            "base_sha256": BASE_SHA256, "data_sha256": DATA_SHA256,
            "data_rows": rows, "expected_steps": EXPECTED_STEPS,
            "gpu": "H200", "max_process_seconds": MAX_PROCESS_SECONDS}


@app.function(cpu=16, memory=32768, timeout=2 * 3600, volumes={"/probe": inputs})
def stage_inputs() -> dict:
    from huggingface_hub import hf_hub_download

    link_inputs()
    model_source.mkdir(parents=True, exist_ok=True)
    for filename in ("model_229067.pt", "meta_229067.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download("castorini/gaggle-nanochat-0.6b-pretrained-climbmix-20261121",
                        filename, revision=BASE_REVISION, local_dir=model_source)
    base = reranker_dir / "data/rlhn250k_prepared/train.parquet"
    if not base.exists():
        subprocess.run(["python", "data_prep/prepare_rlhn.py", "--dataset", "rlhn/rlhn-250K",
                        "--out", "data/rlhn250k_prepared", "--num-proc", "16"],
                       cwd=reranker_dir, check=True)
    if not data_file.exists():
        subprocess.run(["python", "data_prep/relength_nanochat.py",
                        "--src", "data/rlhn250k_prepared/train.parquet",
                        "--out", "data/rlhn250k_prepared_climbmix_lens",
                        "--tokenizer", str(model_dir), "--procs", "16"],
                       cwd=reranker_dir, check=True)
    receipt = check_inputs()
    inputs.commit()
    print("STAGE_OK", json.dumps(receipt), flush=True)
    return receipt


@app.function(cpu=4, memory=16384, timeout=600,
              volumes={"/probe": inputs, "/results": outputs})
def preflight() -> dict:
    receipt = check_inputs()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("ckpt_*.pt")) or (run_dir / "resolved_config.yaml").exists():
        raise RuntimeError(f"Output already contains training state: {run_dir}")
    (run_dir / "preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    outputs.commit()
    print("PREFLIGHT_OK", json.dumps(receipt), flush=True)
    return receipt


@app.function(gpu="H200", cpu=8, memory=32768, timeout=14 * 3600,
              volumes={"/probe": inputs, "/results": outputs},
              single_use_containers=True)
def train() -> dict:
    receipt = check_inputs()
    saved_receipt = json.loads((run_dir / "preflight.json").read_text())
    # Keep the verified inputs pinned while allowing a tighter cap on a resume.
    stable_keys = receipt.keys() - {"max_process_seconds"}
    if any(saved_receipt.get(key) != receipt[key] for key in stable_keys):
        raise RuntimeError("Missing or changed preflight receipt")
    command = [
        "bash", "launch/train.sh", "configs/experiments/gaggle06b_lce.yaml", "1",
        "--override", f"model_name={model_dir}", f"output_dir={run_dir}",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=reranker_dir, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    expired = threading.Event()

    def stop_at_budget() -> None:
        expired.set()
        os.killpg(process.pid, signal.SIGTERM)

    watchdog = threading.Timer(MAX_PROCESS_SECONDS, stop_at_budget)
    watchdog.start()
    saw_budget = saw_plan = False
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("memory:"):
                if "local budget 120,000 tokens" not in line:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise RuntimeError("H200 did not provide the published 120,000-token cap")
                saw_budget = True
            if line.startswith("steps/epoch="):
                match = re.search(r"steps/epoch=([\d,]+) total=([\d,]+) start=(\d+)", line)
                if not match or tuple(int(x.replace(",", "")) for x in match.groups()[:2]) != (EXPECTED_STEPS, EXPECTED_STEPS):
                    os.killpg(process.pid, signal.SIGTERM)
                    raise RuntimeError("Training plan differs from the released 1,928-update recipe")
                saw_plan = True
        code = process.wait()
    finally:
        watchdog.cancel()
    if expired.is_set():
        raise RuntimeError("13-hour attempt budget exhausted; inspect checkpoints and billing before resuming")
    if code != 0 or not (saw_budget and saw_plan):
        raise RuntimeError(f"Trainer ended with code {code}; budget seen={saw_budget}, plan seen={saw_plan}")
    import torch

    final = run_dir / "ckpt_final.pt"
    checkpoint = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
    final_step = checkpoint["step"]
    del checkpoint
    if final_step != EXPECTED_STEPS:
        raise RuntimeError(f"Final checkpoint is at step {final_step}, expected {EXPECTED_STEPS}")
    result = {"status": "complete", "final_step": final_step,
              "wall_seconds": round(time.monotonic() - started, 1),
              "final_checkpoint_bytes": final.stat().st_size, **receipt}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    outputs.commit()
    print("SFT_COMPLETE", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "preflight") -> None:
    if stage == "stage":
        print(json.dumps(stage_inputs.remote(), indent=2))
    elif stage == "preflight":
        print(json.dumps(preflight.remote(), indent=2))
    elif stage == "train":
        call = train.spawn()
        print(json.dumps({"call_id": call.object_id}, indent=2))
    else:
        raise ValueError("stage must be stage, preflight or train")
