from typing import Optional

from .config import (
    OLLAMA_API_KEY,
    OLLAMA_SEARCH_URL,
    SEARCH_MAX_RESULTS,
    SEARCH_SNIPPET_LEN,
    TAVILY_API_KEY,
    TAVILY_SEARCH_URL,
)
from .http_client import client


async def run_ollama_search(query: str) -> list:
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"query": query, "max_results": SEARCH_MAX_RESULTS}
    try:
        r = await client.post(OLLAMA_SEARCH_URL, headers=headers, json=body, timeout=30.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"title": f"Search failed: {e}", "url": "", "content": ""}]

    results = data.get("results", []) if isinstance(data, dict) else []
    out = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if len(content) > SEARCH_SNIPPET_LEN:
            content = content[:SEARCH_SNIPPET_LEN].rstrip() + "..."
        out.append({"title": title, "url": url, "content": content})
    return out


async def run_tavily_search(query: str) -> list:
    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": SEARCH_MAX_RESULTS,
        "include_answer": False,
    }
    try:
        r = await client.post(TAVILY_SEARCH_URL, headers=headers, json=body, timeout=30.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [{"title": f"Search failed: {e}", "url": "", "content": ""}]

    results = data.get("results", []) if isinstance(data, dict) else []
    out = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if len(content) > SEARCH_SNIPPET_LEN:
            content = content[:SEARCH_SNIPPET_LEN].rstrip() + "..."
        out.append({"title": title, "url": url, "content": content})
    return out


SEARCH_BACKENDS = {
    "ollama": run_ollama_search,
    "tavily": run_tavily_search,
}


async def run_search(query: str, backend: Optional[str] = None) -> list:
    """Execute search via a named backend.

    Backends return a list of {"title", "url", "content"} dictionaries so the
    client-specific web_search flows can share the same search provider layer.
    """
    backend_name = (backend or "ollama").strip().lower()
    search = SEARCH_BACKENDS.get(backend_name)
    if search is None:
        return [
            {
                "title": f"Search failed: unknown search backend '{backend_name}'",
                "url": "",
                "content": "",
            }
        ]
    return await search(query)


def _format_search_results_text(hits: list) -> str:
    """Format search hits into a text block for the model to consume."""
    if not hits:
        return "No search results found."
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] {h.get('title', '')}")
        if h.get("url"):
            parts.append(f"    URL: {h['url']}")
        if h.get("content"):
            parts.append(f"    {h['content']}")
    return "\n".join(parts)
