"""Resumable local backfill of Hugging Face documents since 2010.

Docs are stored strictly locally in `data/` (which is gitignored and never committed to Git).
The live Hugging Face Space updates dynamically every week or on demand via Sources / GitHub Actions.

Examples:
    .venv/bin/python scripts/backfill_hf.py --since-year 2010 --device auto
    .venv/bin/python scripts/backfill_hf.py --since-year 2010 --include-arxiv-dataset --max-items 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_news.config import settings  # noqa: E402
from ai_news.fetchers import hf_papers  # noqa: E402
from ai_news.llm import load_embeddings  # noqa: E402
from ai_news.schema import NewsItem  # noqa: E402
from ai_news.store import CorpusStore  # noqa: E402

log = logging.getLogger("backfill_hf")


def _resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _add_chunked(store: CorpusStore, items: list[NewsItem], chunk_size: int) -> int:
    total_added = 0
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        added = store.add(chunk)
        total_added += len(added)
        if added:
            store.save()
            log.info("Checkpoint saved: +%d new (%d/%d processed, corpus=%d)",
                     len(added), min(i + chunk_size, len(items)), len(items), len(store.items))
    return total_added


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill local corpus bank with Hugging Face documents since 2010.")
    p.add_argument("--since-year", type=int, default=settings.corpus_min_year, help="Earliest publication year (default: 2010)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Embedding device (default: auto)")
    p.add_argument("--batch-size", type=int, default=256, help="Embedding batch size (default: 256)")
    p.add_argument("--checkpoint-every", type=int, default=2000, help="Save store to disk every N items (default: 2000)")
    p.add_argument("--workers", type=int, default=6, help="Parallel HTTP workers for HF Daily Papers (default: 6)")
    p.add_argument("--max-items", type=int, default=None, help="Optional cap on fetched items per source")
    p.add_argument("--include-arxiv-dataset", action="store_true",
                   help="Also stream historical arXiv AI/ML metadata dataset from Hugging Face Hub")
    p.add_argument("--arxiv-dataset-id", default="librarian-bots/arxiv-metadata-snapshot",
                   help="HF Hub dataset ID for --include-arxiv-dataset")
    p.add_argument("--push", action="store_true",
                   help="Push backfilled corpus to DATASET_REPO (disabled by default: stays strictly local)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    device = _resolve_device(args.device)
    log.info("Starting local Hugging Face backfill (since=%d, device=%s, data_dir=%s)",
             args.since_year, device, settings.data_dir)

    embeddings = load_embeddings(settings.embed_model_id, device=device, batch_size=args.batch_size)
    store = CorpusStore(
        settings.data_dir,
        embeddings,
        dataset_repo=settings.dataset_repo if args.push else None,
        hf_token=settings.hf_token,
        retention_days=0,  # preserve all historical documents since --since-year
    )
    store.load()
    before = len(store.items)

    status = dict(store.meta.get("source_status") or {})
    fetched: list[NewsItem] = []

    hf_daily = hf_papers.fetch_all_hf_papers(
        since_year=args.since_year, max_items=args.max_items, workers=args.workers
    )
    fetched.extend(hf_daily)
    status["HF Daily Papers"] = f"ok ({len(hf_daily)})"

    hf_blog = hf_papers.fetch_hf_blog_archive(since_year=args.since_year)
    fetched.extend(hf_blog)
    status["Hugging Face Blog"] = f"ok ({len(hf_blog)})"

    if args.include_arxiv_dataset:
        arxiv_hist = hf_papers.fetch_hf_arxiv_dataset(
            since_year=args.since_year, max_items=args.max_items, dataset_id=args.arxiv_dataset_id
        )
        fetched.extend(arxiv_hist)
        status["arXiv (HF Dataset)"] = f"ok ({len(arxiv_hist)})"

    added = _add_chunked(store, fetched, max(100, args.checkpoint_every))
    store.mark_refreshed(status)
    store.save()

    if args.push and store.dataset_repo:
        store.push()

    elapsed = time.time() - t0
    print(f"Backfill complete in {elapsed:.1f}s: fetched {len(fetched):,}, added {added:,} new "
          f"(corpus {before:,} -> {len(store.items):,} items in {settings.data_dir}).")
    for name, n in store.stats()["by_source"].items():
        print(f"  {n:7,d}  {name}")


if __name__ == "__main__":
    main()
