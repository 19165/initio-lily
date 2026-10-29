"""Web search and URL content extraction tool for Lily AI Assistant."""

import re
from typing import Literal

from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import TAVILY_API_KEY, TOOL_DOC_MAX_CHARS
from src.tools.schemas import WebSearchInput
from src.utils import retry_on_network_error

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

_URL_PATTERN = re.compile(r"^https?://[^\s]+", re.IGNORECASE)


@tool(args_schema=WebSearchInput)
@retry_on_network_error(max_retries=3)
def web_search(
    query: str,
    max_results: int = 4,
    topic: Literal["general", "news"] = "general",
) -> str:
    """Search the web for real-time information or extract text from a specific webpage URL.

    Dual-mode behavior:
    1. If `query` is an HTTP/HTTPS URL, extracts and returns the clean text from that webpage.
    2. If `query` is a search phrase, searches Google/Bing via Tavily and returns snippets and URLs.
    """
    if not tavily_client:
        return (
            "Web search is unavailable: TAVILY_API_KEY is not configured in .env. "
            "Please add your Tavily API key to enable live web search."
        )

    clean_query = query.strip()

    # 1. Direct Webpage URL Extraction
    if _URL_PATTERN.match(clean_query):
        try:
            response = tavily_client.extract(urls=[clean_query])
            results = response.get("results", [])
            if not results:
                return f"Could not extract content from URL: {clean_query}"

            raw_text = results[0].get("raw_content", "") or ""
            if not raw_text.strip():
                return f"No readable text content found at URL: {clean_query}"

            truncated = raw_text[:TOOL_DOC_MAX_CHARS]
            if len(raw_text) > TOOL_DOC_MAX_CHARS:
                truncated += "\n...[TRUNCATED TO PREVENT CONTEXT OVERFLOW]"
            return f"📄 Content from {clean_query}:\n\n{truncated}"
        except Exception as e:
            return f"Failed to extract content from {clean_query}: {e!s}"

    # 2. General / News Search
    try:
        clean_topic = "news" if topic == "news" else "general"
        search_res = tavily_client.search(
            query=clean_query,
            search_depth="basic",
            topic=clean_topic,
            max_results=max(1, min(max_results, 10)),
            include_answer=True,
            include_raw_content=False,
        )

        output_parts: list[str] = []

        # Direct AI-synthesized answer from Tavily
        quick_answer = search_res.get("answer")
        if quick_answer:
            output_parts.append(f"💡 Quick Summary: {quick_answer}\n")

        # Result snippets
        items = search_res.get("results", [])
        for item in items:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("content", "").strip()
            output_parts.append(f"🔗 [{title}]({url})\n{snippet}\n")

        if not output_parts:
            return f"No web search results found for query: '{clean_query}'."

        return "\n".join(output_parts).strip()

    except Exception as e:
        return f"Web search failed: {e!s}"


@tool
@retry_on_network_error(max_retries=3)
def search_web(
    query: str,
    max_results: int = 4,
    topic: Literal["general", "news"] = "general",
) -> str:
    """Legacy alias for web_search."""
    return web_search.invoke({"query": query, "max_results": max_results, "topic": topic})


@tool
@retry_on_network_error(max_retries=3)
def fetch_web_page(url: str) -> str:
    """Legacy alias for web_search URL extraction."""
    return web_search.invoke({"query": url})



