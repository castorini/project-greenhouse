"""Compare the released 0.6B reranker with the final Modal SFT checkpoint.

Run after reranker/modal_gaggle06b_sft.py has finished:

    modal run reranker/modal_gaggle06b_compare.py --stage prepare
    modal run reranker/modal_gaggle06b_compare.py --stage probe
    modal run reranker/modal_gaggle06b_compare.py --stage compare
    modal run reranker/modal_gaggle06b_compare.py --stage prepare-beir
    modal run --detach reranker/modal_gaggle06b_compare.py --stage compare-beir

The candidate files stay in a Modal Volume. Only scores, rankings and digests
are written to the comparison result.
"""

import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from itertools import zip_longest
from pathlib import Path

import modal

REPO_COMMIT = "9c878928b89245d411f349e89def7100d18f3611"
BASE_SHA256 = "4e45e1458d00d8d2b869091404ee9c75966431425b972af3364bcd5a5da3ae9d"
RUN_NAME = "gaggle06b_lce_rlhn250k_20261125"
PUBLISHED_REPO = "castorini/gaggle-reranker-0.6b-20261122"
PUBLISHED_REVISION = "1b53dcf2042288af4b0716808f8e3d80341cb597"
CANDIDATES_REPO = "castorini/rank_llm_data"
CANDIDATES_REVISION = "39db0a25a552dd2a12012df22029686bf7fdc050"
QRELS_REVISION = "5a26fd8d1a9e8408ae9a2606e95852312f1e0062"
DL_DATASETS = ("dl19", "dl20", "dl21", "dl22", "dl23")
BEIR_DATASETS = ("covid", "news", "robust04", "nfc", "scifact", "scidocs", "fiqa")
QRELS = {
    **{dataset: f"qrels.{dataset}-passage.txt" for dataset in DL_DATASETS},
    "covid": "qrels.beir-v1.0.0-trec-covid.test.txt",
    "news": "qrels.beir-v1.0.0-trec-news.test.txt",
    "robust04": "qrels.beir-v1.0.0-robust04.test.txt",
    "nfc": "qrels.beir-v1.0.0-nfcorpus.test.txt",
    "scifact": "qrels.beir-v1.0.0-scifact.test.txt",
    "scidocs": "qrels.beir-v1.0.0-scidocs-test.txt",
    "fiqa": "qrels.beir-v1.0.0-fiqa-test.txt",
}
QRELS_SOURCE = {
    "qrels.beir-v1.0.0-scidocs-test.txt": "qrels.beir-v1.0.0-scidocs.test.txt",
    "qrels.beir-v1.0.0-fiqa-test.txt": "qrels.beir-v1.0.0-fiqa.test.txt",
}


def selected_datasets(suite: str) -> tuple[str, ...]:
    if suite == "dl":
        return DL_DATASETS
    if suite == "beir":
        return BEIR_DATASETS
    if suite in BEIR_DATASETS:
        return (suite,)
    raise ValueError(f"Unknown evaluation suite: {suite}")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("git", "build-essential", "openjdk-21-jdk-headless")
    .run_commands(
        "git init /opt/project-greenhouse",
        "git -C /opt/project-greenhouse remote add origin https://github.com/castorini/project-greenhouse.git",
        f"git -C /opt/project-greenhouse fetch --depth 1 origin {REPO_COMMIT}",
        "git -C /opt/project-greenhouse checkout --detach FETCH_HEAD",
        "python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130",
        "python -m pip install -r /opt/project-greenhouse/reranker/requirements.txt",
        "bash /opt/project-greenhouse/pretrain/continue/setup.sh",
    )
    .pip_install("pyserini==2.4.0")
    .run_commands(
        "git clone --depth 1 --branch v9.0.8 https://github.com/usnistgov/trec_eval.git /opt/trec_eval",
        "make -C /opt/trec_eval",
        "make -C /opt/trec_eval quicktest",
    )
    .env({"HF_HOME": "/cache", "PYSERINI_CACHE": "/cache/pyserini",
          "EVAL_TREC_EVAL": "/opt/trec_eval/trec_eval",
          "EVAL_NANOCHAT_SRC": "/opt/project-greenhouse/pretrain/continue/third_party/nanochat",
          "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("greenhouse-gaggle06b-sft-compare", image=image)
cache = modal.Volume.from_name("gaggle-hf-cache", create_if_missing=True)
inputs = modal.Volume.from_name("greenhouse-sft-bounded-probe")
outputs = modal.Volume.from_name("greenhouse-gaggle06b-sft")
repo = Path("/opt/project-greenhouse")
comparison = Path("/results") / RUN_NAME / "comparison"
candidates = comparison / "candidates"
checkpoint = Path("/results") / RUN_NAME / "ckpt_final.pt"
model_source = Path("/probe/models/gaggle-nanochat-0.6b-pretrained-climbmix-20261121")
model_dir = repo / "reranker/models/gaggle-nanochat-0.6b-pretrained-climbmix-20261121"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def same_first_stage_run(dataset: str, trec: list[str]) -> bool:
    reference = repo / "eval/candidates/runs" / f"{dataset}.bm25.trec.gz"
    with gzip.open(reference, "rt") as stream:
        for actual, expected in zip_longest(trec, stream):
            if actual is None or expected is None:
                return False
            a, e = actual.split(), expected.split()
            if (a[0], a[2], a[3]) != (e[0], e[2], e[3]):
                return False
    return True


def check_inputs() -> None:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if not model_dir.exists():
        model_dir.symlink_to(model_source, target_is_directory=True)
    if sha256(model_dir / "model_229067.pt") != BASE_SHA256:
        raise RuntimeError("Backbone SHA256 mismatch")


def verified_inputs(suite: str = "dl") -> dict:
    sys.path.insert(0, str(repo / "eval/candidates"))
    from digest import digest

    expected_candidates = dict(reversed(line.split()) for line in
                               (repo / "eval/candidates/DIGESTS").read_text().splitlines())
    expected_qrels = dict(reversed(line.split()) for line in
                          (repo / "eval/qrels.sha256").read_text().splitlines())
    receipt = {}
    for dataset in selected_datasets(suite):
        path = candidates / f"retrieve_results_{dataset}_top100.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing BM25 candidates: {path}")
        actual_digest, trec = digest(path)
        paper_digest_match = actual_digest == expected_candidates[dataset]
        if dataset in DL_DATASETS and not paper_digest_match:
            raise RuntimeError(f"{dataset} candidates differ from the published input digest")
        if dataset in BEIR_DATASETS and not same_first_stage_run(dataset, trec):
            raise RuntimeError(f"{dataset} candidate IDs differ from the published first-stage run")
        qrels_path = Path("/cache/pyserini/qrels") / QRELS[dataset]
        actual_qrels_hash = sha256(qrels_path)
        if actual_qrels_hash != expected_qrels[QRELS[dataset]]:
            raise RuntimeError(f"{dataset} qrels SHA256 mismatch")
        receipt[dataset] = {"candidate_digest": actual_digest,
                            "paper_digest_match": paper_digest_match,
                            "qrels_sha256": actual_qrels_hash,
                            "queries": sum(1 for _ in path.open())}
    return receipt


@app.function(cpu=2, memory=8192, timeout=1800,
              volumes={"/cache": cache, "/results": outputs})
def prepare(suite: str = "dl") -> dict:
    from huggingface_hub import hf_hub_download

    candidates.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "eval/candidates"))
    from digest import digest
    expected = dict(reversed(line.split()) for line in
                    (repo / "eval/candidates/DIGESTS").read_text().splitlines())
    expected_qrels = dict(reversed(line.split()) for line in
                          (repo / "eval/qrels.sha256").read_text().splitlines())
    qrels_dir = Path("/cache/pyserini/qrels")
    qrels_dir.mkdir(parents=True, exist_ok=True)
    datasets = selected_datasets(suite)
    missing_qrels = [QRELS[dataset] for dataset in datasets
                     if not (qrels_dir / QRELS[dataset]).is_file()
                     or sha256(qrels_dir / QRELS[dataset]) != expected_qrels[QRELS[dataset]]]
    if missing_qrels:
        with tempfile.TemporaryDirectory(dir=qrels_dir) as temp:
            for filename in missing_qrels:
                staged = Path(temp) / filename
                source_name = QRELS_SOURCE.get(filename, filename)
                url = f"https://raw.githubusercontent.com/castorini/eval/{QRELS_REVISION}/qrels/{source_name}"
                urllib.request.urlretrieve(url, staged)
                if sha256(staged) != expected_qrels[filename]:
                    raise RuntimeError(f"{filename} SHA256 differs from the published qrels")
                os.replace(staged, qrels_dir / filename)
        cache.commit()
    for dataset in datasets:
        path = candidates / f"retrieve_results_{dataset}_top100.jsonl"
        try:
            if path.is_file():
                existing_digest, existing_trec = digest(path)
                valid = existing_digest == expected[dataset] or (
                    dataset in BEIR_DATASETS and same_first_stage_run(dataset, existing_trec))
            else:
                valid = False
        except (ValueError, KeyError):
            valid = False
        if valid:
            continue
        with tempfile.TemporaryDirectory(dir=comparison) as temp:
            staged = Path(temp) / path.name
            source = Path(hf_hub_download(CANDIDATES_REPO,
                                          f"retrieve_results/BM25/{path.name}",
                                          repo_type="dataset", revision=CANDIDATES_REVISION))
            if dataset == "dl23":
                # The cache calls this field passage; the evaluator reads contents.
                with source.open() as incoming, staged.open("w") as outgoing:
                    for line in incoming:
                        row = json.loads(line)
                        for candidate in row["candidates"]:
                            candidate["doc"] = {"id": candidate["docid"],
                                                "contents": " ".join(candidate["doc"]["passage"].split())}
                        outgoing.write(json.dumps(row) + "\n")
            else:
                shutil.copy2(source, staged)
            actual_digest, trec = digest(staged)
            if actual_digest != expected[dataset]:
                if dataset in DL_DATASETS or not same_first_stage_run(dataset, trec):
                    raise RuntimeError(f"{dataset} candidate digest and first-stage run differ from published inputs")
                print(f"{dataset}: candidate text differs from the paper digest; first-stage IDs match", flush=True)
            os.replace(staged, path)
            outputs.commit()
            cache.commit()
    receipt = verified_inputs(suite)
    outputs.commit()
    cache.commit()
    print("CANDIDATES_OK", json.dumps(receipt), flush=True)
    return receipt


def card_scores(published_path: str) -> dict:
    import torch

    sys.path.insert(0, str(repo / "eval"))
    from rerank_eval import load_gaggle_hf, load_ours

    query = "what are the symptoms of iron deficiency anemia?"
    passages = [
        "Iron deficiency anemia causes fatigue, shortness of breath, pale skin, and brittle nails.",
        "Iron is a metallic element with atomic number 26, widely used in construction and manufacturing.",
        "A balanced diet includes leafy greens, legumes, and red meat as sources of dietary iron.",
    ]
    published = load_gaggle_hf(published_path, "cuda", torch.bfloat16)
    published_scores = published(query, passages, 3, 512)
    del published
    gc.collect()
    torch.cuda.empty_cache()
    ours = load_ours(str(checkpoint), "cuda", torch.bfloat16)
    ours_scores = ours(query, passages, 3, 512)
    del ours
    gc.collect()
    torch.cuda.empty_cache()
    return {"query": query, "passages": passages,
            "published": published_scores, "reproduced": ours_scores}


def ranks(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        qid, _, docid, rank, score, _ = line.split()
        out.setdefault(qid, []).append((int(rank), docid, float(score)))
    return {qid: sorted(rows) for qid, rows in out.items()}


def agreement(published_path: Path, ours_path: Path) -> dict:
    a, b = ranks(published_path), ranks(ours_path)
    if set(a) != set(b):
        raise RuntimeError("Published and reproduced runs have different query IDs")
    top1 = overlap10 = full_order = rho_sum = 0.0
    for qid in a:
        ar, br = a[qid], b[qid]
        aid = [row[1] for row in ar]
        bid = [row[1] for row in br]
        if set(aid) != set(bid):
            raise RuntimeError(f"Candidate IDs differ for query {qid}")
        n = len(aid)
        top1 += aid[0] == bid[0]
        overlap10 += len(set(aid[:10]) & set(bid[:10])) / min(10, n)
        full_order += aid == bid
        rb = {docid: rank for rank, docid in enumerate(bid, 1)}
        rho_sum += (1.0 if n == 1 else
                    1 - 6 * sum((rank - rb[docid]) ** 2
                                for rank, docid in enumerate(aid, 1)) / (n * (n * n - 1)))
    count = len(a)
    return {"queries": count, "same_top1": int(top1),
            "mean_top10_overlap": round(overlap10 / count, 4),
            "identical_full_rankings": int(full_order),
            "mean_spearman_rho": round(rho_sum / count, 4)}


@app.function(gpu="H200", cpu=8, memory=32768, timeout=5 * 3600,
              volumes={"/probe": inputs, "/cache": cache, "/results": outputs},
              single_use_containers=True)
def compare(limit: int = 0, suite: str = "dl") -> dict:
    from huggingface_hub import snapshot_download
    import torch

    check_inputs()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state["step"] != 1928:
        raise RuntimeError("Comparison requires the final step-1928 checkpoint")
    del state
    receipt = verified_inputs(suite)
    published_path = snapshot_download(PUBLISHED_REPO, revision=PUBLISHED_REVISION,
                                       allow_patterns=["*.json", "*.py", "*.safetensors"])
    tag_suffix = f"probe{limit}" if limit else "full"
    if suite != "dl":
        tag_suffix = f"{suite}_{tag_suffix}"
    result = {"repo_commit": REPO_COMMIT, "published_revision": PUBLISHED_REVISION,
              "final_step": 1928, "limit_queries": limit, "inputs": receipt,
              "datasets": {}}
    if suite == "dl":
        result["card_example"] = card_scores(published_path)
    eval_env = {**os.environ, "EVAL_RUNS": str(candidates),
                "EVAL_QRELS_DIR": "/cache/pyserini/qrels"}
    runs = comparison / tag_suffix / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for dataset in selected_datasets(suite):
        rows = {}
        for backend, model, label in (("gaggle_hf", published_path, "published"),
                                      ("ours", str(checkpoint), "reproduced")):
            tag = f"gaggle06b_{label}_{tag_suffix}"
            saved_result = comparison / tag_suffix / f"{tag}_{dataset}.json"
            saved_run = runs / f"{tag}_{dataset}.trec"
            if saved_result.is_file() and saved_run.is_file():
                rows[label] = json.loads(saved_result.read_text())
                print(f"REUSING {label} {dataset}", flush=True)
                continue
            command = [sys.executable, "eval/rerank_eval.py", "--dataset", dataset,
                       "--backend", backend, "--model", model, "--tag", tag,
                       "--out", str(runs), "--batch-size", "32"]
            if limit:
                command += ["--limit", str(limit)]
            subprocess.run(command, cwd=repo, env=eval_env, check=True)
            result_path = repo / "eval/results" / f"{tag}_{dataset}.json"
            rows[label] = json.loads(result_path.read_text())
            shutil.copy2(result_path, saved_result)
            outputs.commit()
        result["datasets"][dataset] = {
            "published_ndcg10": rows["published"]["ndcg_cut_10"],
            "reproduced_ndcg10": rows["reproduced"]["ndcg_cut_10"],
            "published_seconds": rows["published"]["seconds"],
            "reproduced_seconds": rows["reproduced"]["seconds"],
            "agreement": agreement(runs / f"gaggle06b_published_{tag_suffix}_{dataset}.trec",
                                   runs / f"gaggle06b_reproduced_{tag_suffix}_{dataset}.trec"),
        }
        outputs.commit()
    path = comparison / tag_suffix / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    outputs.commit()
    cache.commit()
    print("COMPARISON_COMPLETE", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(stage: str = "prepare") -> None:
    if stage == "prepare":
        print(json.dumps(prepare.remote(), indent=2))
    elif stage == "probe":
        print(json.dumps(compare.remote(2), indent=2))
    elif stage == "compare":
        print(json.dumps(compare.remote(0), indent=2))
    elif stage.startswith("prepare-"):
        print(json.dumps(prepare.remote(stage.removeprefix("prepare-")), indent=2))
    elif stage == "compare-beir":
        print(json.dumps(compare.remote(0, "beir"), indent=2))
    else:
        raise ValueError("stage must be prepare, probe, compare, prepare-beir or compare-beir")
