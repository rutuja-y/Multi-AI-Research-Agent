
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

        1. Search for relevant information using web_search.
        2. Return clear, relevant search results.
        3. Preserve exact titles and URLs.
        4. Include relevant snippets and URLs.
        5. Never invent URLs or claim to have scraped pages.
        """,
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
        You are a webpage reading and scraping agent.

        1. Select a relevant URL explicitly present in the search results.
        2. Never ask for a URL that has already been provided.
        3. Call scrape_url with the selected URL.
        4. Summarize the retrieved content accurately.
        5. Preserve important facts, dates, statistics, and details.
        6. Include the exact URL attempted.
        7. Clearly report scraping errors or missing content.
        8. Never claim success unless the tool returned page content.
        """,
    )


# Writer Chain

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.

        Write an informative, factual, professional, evidence-based report.
        Avoid repetition and unsupported claims.
        Never invent statistics, citations, or URLs.

        Target length: approximately 1,000-1,500 words.
        Treat this as a target, not a guaranteed exact word count.
        Prioritize accuracy, useful explanations, and relevant evidence."""
    ),
    (
        "human",
        """Write a detailed research report on the following topic.

Topic:
{topic}

Research gathered:
{research}

Use this exact overall structure:

1. Introduction
   Introduce the topic and explain its importance.

2. Key Findings
   Include at least three numbered, well-explained findings.
   Support them with the supplied research.
   Explain relevant evidence, implications, and limitations.

3. Conclusion
   Summarize the main findings without repeating the entire report.

4. Sources
   List relevant URLs explicitly present in the supplied research.
   Do not invent or modify URLs.

Keep the report coherent, factual, professional, and non-repetitive."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# This is called only if the generated report is substantially too long.

shortener_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a careful research editor.
        Shorten reports while preserving important facts and meaning.
        Never invent facts, statistics, or sources.
        Keep the Introduction, Key Findings, Conclusion, and Sources sections.
        Preserve at least three well-explained key findings.
        Keep all relevant source URLs exactly as provided."""
    ),
    (
        "human",
        """Edit the report below to approximately 1,000-1,500 words.

Remove repetition and unnecessary detail, but retain important evidence,
qualifications, explanations, and conclusions.

Return only the revised report.

Report:
{report}"""
    ),
])

shortener_chain = shortener_prompt | llm | StrOutputParser()


# Critic Chain

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a constructive research critic.
        Be honest, specific, and concise.
        Evaluate factual support, clarity, structure, source use,
        and coverage of the topic.
        Keep your complete response under approximately 200 words."""
    ),
    (
        "human",
        """Review the research report below.

Report:
{report}

Respond in this format:

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
