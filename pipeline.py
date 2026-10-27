from agents import build_reader_agent , build_search_agent , writer_chain , critic_chain
import re

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ",state['search_results'])


    # Step 2 - Reader agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    # Extract URLs from the Search Agent's output
    urls = list(dict.fromkeys(
        re.findall(
            r'https?://[^\s<>"\])|]+',
            state["search_results"]
        )
    ))

    if not urls:
        raise RuntimeError(
            "No URLs found in search results. "
            "Cannot proceed with scraping."
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

    # Check whether the scraping tool actually succeeded
    reader_messages = reader_result["messages"]

    successful_scrape = any(
        getattr(message, "type", "") == "tool"
        and "SCRAPE_SUCCESS" in str(message.content)
        for message in reader_messages
    )

    if not successful_scrape:
        raise RuntimeError(
            "Reader Agent did not successfully scrape a webpage. "
            "Stopping the pipeline before report generation."
        )

    state["scraped_content"] = reader_messages[-1].content

    print("\nScraped content:\n", state["scraped_content"])


    #step 3 - writer chain 

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
