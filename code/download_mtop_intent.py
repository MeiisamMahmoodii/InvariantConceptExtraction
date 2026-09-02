"""Download the small public MTOP intent parquet shards used by the replication."""

from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mtop_intent"
BASE = "https://huggingface.co/datasets/mteb/MTOPIntentClassification/resolve/refs%2Fconvert%2Fparquet"


def main():
    for language in ("de", "en", "es", "fr", "hi", "th"):
        for split in ("train", "validation", "test"):
            path = OUT / f"{language}_{split}.parquet"
            if not path.exists():
                OUT.mkdir(parents=True, exist_ok=True)
                urlretrieve(f"{BASE}/{language}/{split}/0000.parquet", path)
            print(f"{path.relative_to(ROOT)}: {path.stat().st_size}")


if __name__ == "__main__":
    main()
