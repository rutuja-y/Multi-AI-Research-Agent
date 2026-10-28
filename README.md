# AI Research Agent System

**Status: Under Development**

An AI-powered research system built with Python that searches the web, extracts relevant information, generates structured research reports, and reviews them using an LLM-based critic.

## Features

- **Web Search:** Finds relevant information using Tavily.
- **Web Scraping:** Extracts webpage content using Requests and BeautifulSoup.
- **AI Research Agents:** Uses LangChain and Groq to coordinate research and report generation.
- **Report Generation:** Produces structured reports targeting approximately 1,000–1,500 words.
- **Critic Review:** Evaluates generated reports and provides feedback.
- **Streamlit Interface:** Allows users to enter research topics, view reports and feedback, monitor progress, and download reports as Markdown files.

## Tech Stack

- Python
- LangChain
- Groq API
- Tavily Search
- Requests and BeautifulSoup
- Streamlit
- python-dotenv

## Project Structure

```text
AI-Research-Agent/
├── agents.py       # Agents and LLM chains
├── tools.py        # Web search and scraping tools
├── pipeline.py     # Research workflow and error handling
├── app.py          # Streamlit application
├── requirements.txt
├── .env            # API keys
└── README.md
```

## Setup and Run

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add the required API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## Future Improvements

- [ ] Multi-source research with evidence-grounded citations.
- [ ] Automated testing with pytest.
- [ ] Research quality evaluation and benchmarking.
- [ ] RAG support for uploaded documents.
- [ ] FastAPI backend for the research pipeline.
- [ ] Improved documentation, Docker support, and deployment.

## Project Goal

Build a reliable and extensible AI research system while exploring LLM integration, agent workflows, web data extraction, evaluation, and backend development.

_This project is actively under development, and features and architecture may evolve as improvements are implemented._
