import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sync_readme import synced_readme  # noqa: E402

ABOUT = (ROOT / "ai_news" / "about.md").read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"[-_\s]+", " ", text.lower())


def test_about_explains_every_dependency():
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[ ]", line)[0])
    # preinstalled on Spaces, so not in requirements.txt, but still part of the stack
    names += ["gradio", "spaces", "huggingface_hub", "langgraph", "langchain-core"]
    missing = [n for n in names if _norm(n) not in _norm(ABOUT)]
    assert not missing, f"about.md does not explain: {missing}"


def test_readme_section_matches_about():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert synced_readme(readme, ABOUT) == readme, "README is stale: run `python scripts/sync_readme.py`"


def test_about_tab_is_in_the_ui(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from ai_news.store import CorpusStore
    from ai_news.ui import build_ui
    demo = build_ui(CorpusStore(tmp_path, DeterministicFakeEmbedding(size=8)), ["arXiv"],
                    lambda *a: None, lambda: "")
    labels = [b.label for b in demo.blocks.values() if type(b).__name__ == "Tab"]
    assert labels == ["Ask", "Search", "Latest", "Sources", "About"]
