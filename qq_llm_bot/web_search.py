from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from loguru import logger

from qq_llm_bot.config import LexiconConfig


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchClient:
    def __init__(self, config: LexiconConfig) -> None:
        self.config = config

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        provider = self.config.provider.strip().lower()
        if not self.config.enabled or provider in {"", "disabled", "none"}:
            return []

        limit = max(1, min(max_results or self.config.max_results, self.config.max_results))
        try:
            if provider == "duckduckgo":
                return await self._search_duckduckgo(query, limit)
            if provider == "serper":
                return await self._search_serper(query, limit)
            if provider == "searxng":
                return await self._search_searxng(query, limit)
        except httpx.HTTPError as exc:
            logger.warning("Web search failed: {}", exc)
            return []

        logger.warning("Unknown web search provider: {}", self.config.provider)
        return []

    async def _search_duckduckgo(self, query: str, limit: int) -> list[SearchResult]:
        url = self.config.base_url.strip() or "https://duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0 qq-llm-bot/0.1"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url, params={"q": query}, headers=headers)
            response.raise_for_status()
        return _parse_duckduckgo_html(response.text, limit)

    async def _search_serper(self, query: str, limit: int) -> list[SearchResult]:
        api_key = resolve_search_api_key(self.config)
        if not api_key:
            logger.warning("Serper search is enabled but no API key was configured")
            return []

        url = self.config.base_url.strip() or "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json={"q": query, "num": limit})
            response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic", [])[:limit]:
            title = str(item.get("title", "")).strip()
            link = str(item.get("link", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results

    async def _search_searxng(self, query: str, limit: int) -> list[SearchResult]:
        if not self.config.base_url:
            logger.warning("SearXNG search is enabled but lexicon.base_url is empty")
            return []

        url = urljoin(self.config.base_url.rstrip("/") + "/", "search")
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.get(url, params={"q": query, "format": "json"})
            response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", [])[:limit]:
            title = str(item.get("title", "")).strip()
            link = str(item.get("url", "")).strip()
            snippet = str(item.get("content", "")).strip()
            if title and link:
                results.append(SearchResult(title=title, url=link, snippet=snippet))
        return results


def build_web_search_client(config: LexiconConfig) -> WebSearchClient:
    return WebSearchClient(config)


def resolve_search_api_key(config: LexiconConfig) -> str:
    return config.api_key or os.getenv(config.api_key_env, "")


def _parse_duckduckgo_html(body: str, limit: int) -> list[SearchResult]:
    links = list(
        re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            re.S,
        )
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|'
        r'<div[^>]+class="result__snippet"[^>]*>(.*?)</div>',
        body,
        re.S,
    )

    results: list[SearchResult] = []
    for index, match in enumerate(links[:limit]):
        href = _normalize_duckduckgo_url(html.unescape(match.group(1)))
        title = _strip_html(match.group(2))
        snippet_raw = ""
        if index < len(snippets):
            snippet_raw = next((part for part in snippets[index] if part), "")
        snippet = _strip_html(snippet_raw)
        if title and href:
            results.append(SearchResult(title=title, url=href, snippet=snippet))
    return results


def _normalize_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    if url.startswith("/"):
        return "https://duckduckgo.com" + url
    return url


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return " ".join(text.split())


def default_slang_query(term: str) -> str:
    return f"{term} 网络用语 梗 意思"
