"""Bootstrap or refresh the corpus from the command line.

    python scripts/build_corpus.py --days 30                # fetch, index, push to $DATASET_REPO
    python scripts/build_corpus.py --days 7 --no-push       # local only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_news.config import settings  # noqa: E402
from ai_news.ingest import refresh  # noqa: E402
from ai_news.llm import load_embeddings  # noqa: E402
from ai_news.store import CorpusStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=settings.lookback_days)
    p.add_argument("--dataset-repo", default=settings.dataset_repo)
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    store = CorpusStore(settings.data_dir, load_embeddings(settings.embed_model_id),
                        dataset_repo=args.dataset_repo, hf_token=settings.hf_token)
    store.pull()
    store.load()
    print(refresh(store, settings.sources_file, args.days, push=not args.no_push))
    for name, n in store.stats()["by_source"].items():
        print(f"  {n:6d}  {name}")
    for name, st in store.meta.get("source_status", {}).items():
        if not st.startswith("ok"):
            print(f"  FAILED {name}: {st}")


if __name__ == "__main__":
    main()
