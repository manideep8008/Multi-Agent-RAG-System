"""
tools/web_search.py - Wikipedia search tool.

This tool searches Wikipedia for supplementary information when
local course documents don't fully answer a question.
Maps to Paper 1 (Agentic RAG) Section 4.2: Web Search Agent.

Uses Wikipedia's free API - no API key needed.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request
import urllib.parse
import json


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search Wikipedia for supplementary information.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 3)

    Returns:
        Formatted string with Wikipedia search results
    """
    try:
        # Step 1: Search for matching articles
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1,
            })
        )

        req = urllib.request.Request(search_url, headers={"User-Agent": "CourseQABot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            search_data = json.loads(resp.read().decode())

        results = search_data.get("query", {}).get("search", [])

        if not results:
            return "No Wikipedia results found for this query."

        # Step 2: Get summaries for each result
        page_titles = [r["title"] for r in results]
        summary_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "titles": "|".join(page_titles),
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exlimit": max_results,
                "format": "json",
                "utf8": 1,
            })
        )

        req = urllib.request.Request(summary_url, headers={"User-Agent": "CourseQABot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            summary_data = json.loads(resp.read().decode())

        pages = summary_data.get("query", {}).get("pages", {})

        output = []
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            title = page.get("title", "Unknown")
            extract = page.get("extract", "No summary available.")
            # Truncate long extracts
            if len(extract) > 800:
                extract = extract[:800] + "..."
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            output.append(f"[{title}]\n{extract}\nURL: {url}")

        if not output:
            return "No Wikipedia results found for this query."

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


# Quick test
if __name__ == "__main__":
    print("Testing Wikipedia search...")
    # print(search_web("star schema data warehousing"))
    # print("\n" + "=" * 60 + "\n")
    # print(search_web("Indian Premier League 2025"))