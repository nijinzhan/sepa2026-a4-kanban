"""Fetch pinned public research assets; never execute the downloaded scripts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import HfApi, hf_hub_download
from whisper_atc_experiment_2024141460331 import HERE, MODELS, RESEARCH, sha256, write_json

CODE_REVISION = "cd0b71e5e0da9241d589d4bdbfbf16440549e577"
DATA_REVISION = "3b96f5791582cf6f6f348ca46fd502a981918d85"
DATA_FILES = {
    "data/train-00000-of-00001-4907384ef649141a.parquet": (
        "16ca19cc381078cfdf2f6a8c3161a76ffed9653520a7ed89d7da5bb03edf88a0"
    ),
    "data/validation-00000-of-00001-b20d1116e523bc85.parquet": (
        "9c345ebf0ff60c6b7b6f79063dd615e7d0e051343f4c04465590ed24f29e159c"
    ),
}


def download_model(info):
    local = HERE / "models" / info["id"].split("/")[-1]
    files = HfApi().list_repo_files(info["id"], revision=info["revision"])
    for name in files:
        if "/" not in name and Path(name).suffix in (".json", ".txt", ".md", ".safetensors"):
            hf_hub_download(info["id"], name, revision=info["revision"], local_dir=local)
    if sha256(local / "model.safetensors") != info["sha256"]:
        raise ValueError(f"model SHA-256 mismatch: {info['id']}")
    print("VERIFIED", info["id"], flush=True)
    return info


def main():
    provenance = {"codeRevision": CODE_REVISION, "datasetRevision": DATA_REVISION, "files": {}}
    for name in (
        "README.md",
        "LICENSE.txt",
        "Evaluate/Normalizer.py",
        "Evaluate/Evaluate-FineTuned.py",
        "Evaluate/Evaluate-Blank.py",
        "CreateDataset/01-Prepare.py",
        "Fine-Tune/FineTuning-ATCO2-ATCOSIM.py",
    ):
        url = f"https://raw.githubusercontent.com/jlvdoorn/WhisperATC/{CODE_REVISION}/{name}"
        target = RESEARCH / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=90) as response:
            content = response.read()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"existing pinned source differs: {name}")
        target.write_bytes(content)
        provenance["files"][name] = {"url": url, "sha256": sha256(target)}
    for name, expected in DATA_FILES.items():
        target = hf_hub_download(
            "jlvdoorn/atco2-asr",
            name,
            repo_type="dataset",
            revision=DATA_REVISION,
            local_dir=RESEARCH / "published-atco2-dataset",
        )
        if sha256(Path(target)) != expected:
            raise ValueError(f"dataset SHA-256 mismatch: {name}")
        provenance["files"][name] = {"sha256": expected}
    with ThreadPoolExecutor(max_workers=2) as pool:
        provenance["models"] = list(pool.map(download_model, MODELS.values()))
    write_json(RESEARCH / "asset-provenance.json", provenance)


if __name__ == "__main__":
    main()
