"""End-to-end smoke test against a running app (local URL or a Space id).

    python scripts/smoke_test.py http://127.0.0.1:7860
    python scripts/smoke_test.py <you>/ai-news-navigator
"""

from __future__ import annotations

import os
import re
import sys

from gradio_client import Client


def main(target: str) -> None:
    c = Client(target, token=os.getenv("HF_TOKEN"), httpx_kwargs={"timeout": 600}, verbose=False)
    api = c.view_api(print_info=False, return_format="dict")
    print("endpoints:", sorted(api["named_endpoints"]))

    status = c.predict(api_name="/status")
    print("status:", status.splitlines()[0])

    html = c.predict("language model reasoning", [], "any", "Any time", "relevance", api_name="/search")
    urls = re.findall(r'href="(https?://[^"]+)"', html)
    print(f"search: {len(urls)} links, e.g. {urls[:2]}")
    assert urls, "keyword search returned no links"

    job = c.submit("What are recent approaches to improving LLM reasoning?", [], "any", "Any time",
                   api_name="/research_assistant")
    answer, refs, sieve, _ = job.result()
    assert "stopped with an error" not in answer and "GPU time" not in answer, f"assistant failed: {answer}"
    ref_urls = re.findall(r'href="(https?://[^"]+)"', refs)
    print("sieve counts:", re.findall(r'class="count">([^<]+)<', sieve))
    print("answer:", str(answer)[:600])
    print(f"references: {len(ref_urls)}", ref_urls[:3])
    assert ref_urls, "assistant returned no references"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860")
