"""Copy ai_news/about.md (the About tab) into README.md between the about markers.

    python scripts/sync_readme.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- about:start -->", "<!-- about:end -->"


def synced_readme(readme: str, about: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit(f"README.md needs the markers {START} and {END}")
    block = f"{START}\n{about.strip()}\n{END}"
    return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, readme, flags=re.S)


def main() -> int:
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    new = synced_readme(readme, (ROOT / "ai_news" / "about.md").read_text(encoding="utf-8"))
    if new != readme:
        readme_path.write_text(new, encoding="utf-8")
        print("README.md updated from ai_news/about.md")
    else:
        print("README.md already in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
