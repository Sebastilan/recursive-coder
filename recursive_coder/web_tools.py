"""Web search and page fetch tools for agent use."""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import httpx

from .logger_setup import get_logger

logger = get_logger("web_tools")

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# ── Proxy resolution ─────────────────────────────────────────────────────────

def _get_proxy(proxy_override: str | None = None) -> str | None:
    """Resolve proxy: explicit > env var > None."""
    if proxy_override:
        return proxy_override
    return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None


# ── HTML-to-text converter (stdlib only) ──────────────────────────────────────

class _HTMLToText(HTMLParser):
    """Minimal HTML to plain text converter."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "path", "head"}

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip_depth = 0
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "pre":
            self._in_pre = True
        elif tag in ("br", "hr"):
            self._text.append("\n")
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "pre":
            self._in_pre = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "table"):
            self._text.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        self._text.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text)
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)


def html_to_text(html_content: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLToText()
    try:
        parser.feed(html_content)
    except Exception:
        pass
    return parser.get_text()


# ── DuckDuckGo HTML search result parser ─────────────────────────────────────

def _parse_ddg_results(html_content: str) -> list[dict]:
    """Parse DuckDuckGo HTML search results.

    DuckDuckGo HTML version (html.duckduckgo.com) returns server-rendered results.
    Result structure:
        <a class="result__a" href="//duckduckgo.com/l/?uddg=ENCODED_URL&...">TITLE</a>
        <a class="result__snippet" ...>SNIPPET</a>
    """
    results = []

    # Extract title + URL pairs
    title_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    # Extract snippets
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    titles = title_pattern.findall(html_content)
    snippets = snippet_pattern.findall(html_content)

    for i, (href, title_html) in enumerate(titles):
        # Decode DDG redirect URL
        if "uddg=" in href:
            parsed = parse_qs(urlparse(href).query)
            url = unquote(parsed.get("uddg", [""])[0])
        else:
            url = href
            if url.startswith("//"):
                url = "https:" + url

        title = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        if url.startswith("http") and title:
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet[:300],
            })

    return results


# ── Public API ────────────────────────────────────────────────────────────────

async def web_search(
    query: str,
    max_results: int = 5,
    proxy: str | None = None,
) -> str:
    """Search the web using DuckDuckGo HTML and return formatted results.

    Returns:
        [1] Title
            URL: https://...
            Snippet text...
    """
    effective_proxy = _get_proxy(proxy)
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    headers = {
        "User-Agent": _DEFAULT_UA,
        "Accept": "text/html",
        "Cookie": "kl=us-en",  # force English results
    }

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            proxy=effective_proxy,
            follow_redirects=True,
        ) as client:
            resp = await client.get(search_url, headers=headers)

        if resp.status_code != 200:
            return f"ERROR: Search returned status {resp.status_code}"

        results = _parse_ddg_results(resp.text)
        if not results:
            logger.warning("No results parsed from DDG HTML (query=%s)", query)
            return f"No search results found for: {query}"

        results = results[:max_results]
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            if r.get("snippet"):
                lines.append(f"    {r['snippet']}")
            lines.append("")

        logger.info("web_search query=%r results=%d proxy=%s", query, len(results), bool(effective_proxy))
        return "\n".join(lines).strip()

    except httpx.TimeoutException:
        return "ERROR: Search request timed out (15s)"
    except Exception as exc:
        return f"ERROR: Search failed: {exc}"


async def fetch_page(
    url: str,
    max_chars: int = 8000,
    proxy: str | None = None,
) -> str:
    """Fetch a web page and return its text content.

    HTML is converted to readable plain text. Output is truncated to max_chars.
    """
    effective_proxy = _get_proxy(proxy)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"ERROR: Invalid URL scheme: {parsed.scheme}"

    headers = {
        "User-Agent": _DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            proxy=effective_proxy,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code != 200:
            return f"ERROR: HTTP {resp.status_code} for {url}"

        content_type = resp.headers.get("content-type", "")

        # Non-HTML content: return raw text (truncated)
        if "html" not in content_type and "xml" not in content_type:
            text = resp.text[:max_chars]
            if len(resp.text) > max_chars:
                text += f"\n... [truncated, {len(resp.text)} chars total]"
            return text

        # HTML: convert to text
        text = html_to_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

        logger.info("fetch_page url=%s chars=%d proxy=%s", url, len(text), bool(effective_proxy))
        return text if text.strip() else "(page has no readable text content)"

    except httpx.TimeoutException:
        return f"ERROR: Request timed out (20s) for {url}"
    except Exception as exc:
        return f"ERROR: Failed to fetch {url}: {exc}"
