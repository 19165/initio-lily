from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import TAVILY_API_KEY, TOOL_DOC_MAX_CHARS
from src.utils import retry_on_network_error

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


@tool
@retry_on_network_error(max_retries=3)
def search_web(
    query: str,
    topic: str = "general",
    max_results: int = 4,
) -> str:
    """Search the web for real-time information, news, facts, weather, or technical documentation.

    Args:
        query: Specific search keywords (e.g. 'Bitcoin price today', 'Bangkok weather', 'Python 3.13 changelog').
        topic: Choose 'general' for general queries/facts, or 'news' for recent breaking news.
        max_results: Number of search results to fetch (default: 4).
    """
    if not tavily_client:
        return "Tavily API key is missing. Please configure TAVILY_API_KEY in .env."

    try:
        clean_topic = (
            topic.strip().lower() if topic in ["general", "news"] else "general"
        )
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            topic=clean_topic,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )

        results = []

        # 1. Direct answer summary from Tavily
        if response.get("answer"):
            results.append(f"💡 Quick Summary: {response['answer']}\n")

        # 2. Detailed search result snippets with URLs
        raw_results = response.get("results", [])
        for item in raw_results:
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "").strip()
            results.append(f"🔗 [{title}]({url})\n{content}\n")

        if not results:
            return "No relevant search results found."

        return "\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
@retry_on_network_error(max_retries=3)
def fetch_web_page(url: str) -> str:
    """Fetch and extract clean readable text from a specific web page URL using Tavily Extract.

    Args:
        url: The full HTTP/HTTPS URL of the web page.
    """
    if not tavily_client:
        return "Tavily API key is missing. Please configure TAVILY_API_KEY in .env."

    try:
        response = tavily_client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return "Failed to extract content from URL."

        text = results[0].get("raw_content", "")
        if not text:
            return "No content could be extracted from this URL."

        return text[:TOOL_DOC_MAX_CHARS] + (
            "\n...[TRUNCATED]" if len(text) > TOOL_DOC_MAX_CHARS else ""
        )
    except Exception as e:
        return f"Failed to fetch URL: {str(e)}"
