from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
from rich import print


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    results = tavily.search(query=query,max_results=5)
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL:{r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    return "\n----\n".join(out)
    


@tool
def scrape_url(url: str) -> str:
    """Fetch a webpage URL and return its cleaned text content."""

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript"
        ]):
            tag.decompose()

        content = soup.get_text(separator=" ", strip=True)

        if not content:
            return f"SCRAPE_ERROR: No readable text found at {url}"

        return (
            f"SCRAPE_SUCCESS\n"
            f"URL: {url}\n"
            f"PAGE CONTENT:\n{content[:10000]}"
        )

    except requests.RequestException as exc:
        return f"SCRAPE_ERROR: Could not fetch {url}: {exc}"

    except Exception as exc:
        return f"SCRAPE_ERROR: Failed to process {url}: {exc}"

