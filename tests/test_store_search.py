from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_news.search import highlight, keyword_search
from ai_news.store import CorpusStore
from conftest import make_item


def test_add_dedupes_and_merges_upvotes(store):
    n = len(store.items)
    dup = make_item("Test-time compute scaling for reasoning models", url=next(
        it.url for it in store.items.values() if it.title.startswith("Test-time")), upvotes=42)
    assert store.add([dup]) == []
    assert len(store.items) == n
    assert max(it.upvotes for it in store.items.values()) == 42


def test_bm25_and_vector_search(store):
    hits = store.bm25_scores("test-time compute reasoning")
    assert hits[0][0].title.startswith("Test-time")
    assert len(store.vector_search("anything", k=3)) == 3


def test_keyword_search_filters(store):
    assert [h[0].source_type for h in keyword_search(store, "model", source_type="blog")] == ["blog"]
    recent = keyword_search(store, "", days=7)
    assert {h[0].title for h in recent} == {"Test-time compute scaling for reasoning models",
                                             "Introducing our new open-weight multimodal model"}
    assert keyword_search(store, '"open weights"')[0][0].source == "Hugging Face Blog"
    assert keyword_search(store, '"no such phrase anywhere"') == []


def test_save_load_roundtrip_and_prune(store, tmp_path):
    store.save()
    fresh = CorpusStore(tmp_path, DeterministicFakeEmbedding(size=32), retention_days=180)
    fresh.load()
    assert set(fresh.items) == set(store.items)
    assert fresh.prune() == 1  # the 200-day-old item
    assert len(fresh.vectors) == len(fresh.items)


def test_highlight_escapes_html():
    out = highlight("<script> RAG survey", "rag")
    assert "&lt;script&gt;" in out and "<mark>RAG</mark>" in out


def test_reload_if_stale_picks_up_other_process_writes(store, tmp_path):
    import os
    store.save()
    reader = CorpusStore(tmp_path, DeterministicFakeEmbedding(size=32))
    reader.load()
    assert reader.reload_if_stale() is False
    store.add([make_item("Brand new paper on world models", "World models for robotics.")])
    store.save()
    meta = tmp_path / "meta.json"
    os.utime(meta, (meta.stat().st_atime, meta.stat().st_mtime + 5))  # make sure the mtime advances
    assert reader.reload_if_stale() is True
    assert len(reader.items) == len(store.items)


def test_sync_from_hub_reloads_only_when_snapshot_changes(store, tmp_path):
    import shutil

    from ai_news.ingest import sync_from_hub

    remote = tmp_path / "remote"   # stands in for the HF Dataset repo
    store.data_dir = remote
    store.mark_refreshed({"arXiv": "ok (1)"})
    store.save()

    space = CorpusStore(tmp_path / "space", DeterministicFakeEmbedding(size=32), dataset_repo="me/corpus")

    def fake_pull():
        shutil.copytree(remote, space.data_dir, dirs_exist_ok=True)
        return True

    space.pull = fake_pull
    assert "Pulled new snapshot" in sync_from_hub(space)
    assert set(space.items) == set(store.items)
    assert "Already up to date" in sync_from_hub(space)

    store.add([make_item("A newer paper on agents", "Tool-using agents.")])
    store.mark_refreshed({"arXiv": "ok (2)"})
    store.save()
    assert "Pulled new snapshot" in sync_from_hub(space)
    assert len(space.items) == len(store.items)
