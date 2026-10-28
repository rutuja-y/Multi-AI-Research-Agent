
from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    shortener_chain,
)
import re
import time


def count_words(text: str) -> int:
    """Count words without requiring an additional dependency."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def review_with_retry(report: str) -> str:
    """Retry temporary rate-limit failures without rerunning research."""

    for attempt in range(3):
        try:
            return critic_chain.invoke({"report": report})

        except Exception as exc:
            error_text = str(exc).lower()

            is_rate_limit = (
                "429" in error_text
                or "rate limit" in error_text
                or "rate_limit" in error_text
                or "tokens per minute" in error_text
            )

            # Retry only rate-limit errors, not unrelated failures.
            if not is_rate_limit:
                return (
                    "Critic review failed. The research report was preserved. "
                    f"Error: {str(exc)[:300]}"
                )

            # Three total attempts: initial attempt plus two retries.
            if attempt == 2:
                return (
                    "Critic review could not be completed after 3 attempts "
                    "because of a Groq rate limit. The research report "
                    "was preserved. Try reviewing it again later."
                )

            wait_seconds = 2 ** (attempt + 1)
            print(
                f"Critic rate limit reached. "
                f"Retrying in {wait_seconds} seconds "
                f"(retry {attempt + 1}/2)..."
            )
            time.sleep(wait_seconds)

    return "Critic review could not be completed. The report was preserved."


def run_research_pipeline(topic: str, progress_callback=None) -> dict:

    def notify(stage: str) -> None:
        """Tell the UI which stage is starting (does nothing if no callback)."""
        if progress_callback:
            progress_callback(stage)

    state = {}

    # Step 1 - Search agent
    notify("search")
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            ("user", f"Find recent, reliable and detailed information about: {topic}")
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSearch results:\n", state["search_results"])

    # Step 2 - Reader agent
    notify("scrape")
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    urls = list(dict.fromkeys(
        re.findall(
            r'https?://[^\s<>"\])}]+',
            state["search_results"]
        )
    ))

    if not urls:
        raise RuntimeError(
            "No URLs found in search results. Cannot proceed with scraping."
        )

    print("URLs found:", urls)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"Research topic: {topic}\n\n"
                "Select the most relevant URL from this list and call "
                "the scrape_url tool to retrieve its content.\n\n"
                f"Available URLs:\n{chr(10).join(urls[:5])}\n\n"
                "Do not ask me for a URL. Use one of the URLs above."
            )
        ]
    })

    reader_messages = reader_result["messages"]

    successful_scrape = any(
        getattr(message, "type", "") == "tool"
        and "SCRAPE_SUCCESS" in str(message.content)
        for message in reader_messages
    )

    if not successful_scrape:
        raise RuntimeError(
            "Reader Agent did not successfully scrape a webpage. "
            "Stopping before report generation."
        )

    state["scraped_content"] = reader_messages[-1].content

    print("\nScraped content:\n", state["scraped_content"])

    # Step 3 - Writer chain
    notify("write")
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    original_word_count = count_words(state["report"])
    print(f"\nGenerated report word count: {original_word_count}")

    # Shorten only if substantially above the desired range.
    # Keep the original report if the editing request fails.
    if original_word_count > 1650:
        print("Report is too long. Attempting one shortening pass...")

        try:
            shortened_report = shortener_chain.invoke({
                "report": state["report"]
            })

            if shortened_report.strip():
                state["report"] = shortened_report
                print(
                    "Shortened report word count:",
                    count_words(state["report"]),
                )
            else:
                print("Shortening returned an empty report; keeping original.")
        except Exception as exc:
            print(
                "Shortening failed; keeping original report:",
                str(exc)[:300],
            )

    final_word_count = count_words(state["report"])
    state["report_word_count"] = final_word_count

    if not 1000 <= final_word_count <= 1500:
        print(
            f"Note: Report has {final_word_count} words. "
            "The target is approximately 1,000-1,500 words."
        )

    print("\nFinal Report:\n", state["report"])

    # Step 4 - Critic chain
    notify("critic")
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    # This step never reruns search, scraping, or writing.
    state["feedback"] = review_with_retry(state["report"])

    print("\nCritic report:\n", state["feedback"])


    notify("done")
    return state



if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    result = run_research_pipeline(topic)

    print("\nPipeline completed.")
    print("Report words:", result.get("report_word_count", "Unknown"))
