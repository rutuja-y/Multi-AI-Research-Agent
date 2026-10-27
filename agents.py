
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
        You are a web research search agent.

        For every research request:
        1. Use the web_search tool to search for relevant information.
        2. Return the search results clearly.
        3. Preserve the exact titles and URLs returned by the tool.
        4. Include the URLs and relevant snippets in your final response.
        5. Do not invent URLs or claim to have scraped webpages.
        """,
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are a webpage reading and scraping agent.

        You will receive a research topic and search results containing URLs.

        Follow these instructions:
        1. Identify the most relevant URL explicitly present in the
           provided search results.
        2. Never ask the user to provide a URL if one is already present.
        3. Call the scrape_url tool with the selected URL.
        4. Read the content returned by the tool.
        5. Summarize the retrieved content, preserving important facts,
           dates, statistics, and relevant details.
        6. Include the exact URL you attempted to scrape.
        7. If the tool returns an error or no useful page content,
           clearly report the scraping failure.
        8. Never claim you scraped a page unless the tool returned
           its content.
        """,
    )


# Writer Chain (LCEL)

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.
        Write clear, structured, insightful and evidence-based reports.
        Distinguish verified source information from unsupported claims.
        Do not invent statistics, citations, or URLs."""
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list the URLs actually present in the research)

Be detailed, factual and professional."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# Critic Chain (LCEL)

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line Verdict:
- ..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
