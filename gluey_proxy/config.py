import os


UPSTREAM = os.environ.get("UPSTREAM", "http://litellm:4000")
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
LOG_REQUESTS = os.environ.get("LOG_REQUESTS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
LOG_BODY_TO_STDOUT = os.environ.get("LOG_BODY_TO_STDOUT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
LOG_BODY_MAX_CHARS = int(os.environ.get("LOG_BODY_MAX_CHARS", "20000"))

OLLAMA_SEARCH_URL = "https://ollama.com/api/web_search"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
SEARCH_SNIPPET_LEN = int(os.environ.get("SEARCH_SNIPPET_LEN", "1500"))

# Web search backends: "ollama" (default) or "tavily".
CLAUDE_SEARCH_BACKEND = os.environ.get("CLAUDE_SEARCH_BACKEND", "ollama").strip().lower()
CODEX_SEARCH_BACKEND = os.environ.get("CODEX_SEARCH_BACKEND", "ollama").strip().lower()

# Priority order used when a search backend is set to "auto". The first
# configured backend that returns results wins; failures fall back to the next.
SEARCH_AUTO_ORDER = [
    name.strip().lower()
    for name in os.environ.get("SEARCH_AUTO_ORDER", "ollama,tavily,searxng").split(",")
    if name.strip()
]

# When enabled, inject a web_search function tool into /v1/responses requests
# that don't already carry one. Lets clients whose web search is not exposed to
# the model (e.g. Kilo on a custom provider) use the proxy's search backend.
INJECT_WEBSEARCH = os.environ.get("INJECT_WEBSEARCH", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "").strip().rstrip("/")

WEB_SEARCH_SYSTEM_MARKER = "You are an assistant for performing a web search tool use"
WEB_SEARCH_QUERY_PREFIX = "Perform a web search for the query: "
