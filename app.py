import re

import streamlit as st

from pipeline import run_research_pipeline

# ---------- Page setup ----------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

# Each stage: (key sent by the pipeline, label, short description)
STAGES = [
    ("search", "Search", "Finding recent, reliable sources"),
    ("scrape", "Scraping", "Reading the most relevant web page"),
    ("write", "Writing", "Drafting the research report"),
    ("critic", "Critic Review", "Reviewing the draft"),
]


# ---------- Helper functions ----------
def render_progress(placeholder, current, failed=False):
    """Draw the progress bar and the stage checklist inside a placeholder."""
    keys = [stage[0] for stage in STAGES]
    index = len(STAGES) if current == "done" else keys.index(current)

    lines = []
    for i, (_, label, description) in enumerate(STAGES):
        if i < index:
            icon = "✅"
        elif i == index:
            icon = "❌" if failed else "⏳"
        else:
            icon = "⚪"
        lines.append(f"{icon} **{label}**: {description}")

    with placeholder.container():
        st.progress(index / len(STAGES))
        st.markdown("\n\n".join(lines))


def make_filename(topic):
    """Turn a topic into a safe file name, e.g. 'my-topic.md'."""
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return f"{slug[:50] or 'research-report'}.md"


def count_words(text):
    """Same word-counting rule as pipeline.py (used only as a fallback)."""
    return len(re.findall(r"\b[\w'-]+\b", text))


# ---------- Header ----------
st.title("🔎 AI Research Agent")
st.caption("Enter a topic and the agents will search, read, write and review a report for you.")
st.divider()

# ---------- Input ----------
topic = st.text_input(
    "Research topic",
    placeholder="e.g. The impact of solid-state batteries on electric vehicles",
)
run_clicked = st.button("Generate Report", type="primary", use_container_width=True)

# ---------- Run the pipeline ----------
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        # Clear results from any previous run
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)

        progress_box = st.empty()
        current = {"stage": "search"}  # remembers where we are, in case of an error
        
        def on_progress(stage):
            current["stage"] = stage

            if stage == "done":
                return

            render_progress(progress_box, stage)

        try:
            result = run_research_pipeline(
                topic.strip(),
                progress_callback=on_progress,
            )

            st.session_state["result"] = result
            st.session_state["topic"] = topic.strip()

            feedback = str(result.get("feedback", ""))

            if feedback.startswith("Critic review"):
                render_progress(progress_box, "critic", failed=True)
            else:
                render_progress(progress_box, "done")
                
        except Exception as exc:
            render_progress(progress_box, current["stage"], failed=True)
            st.session_state["error"] = (
                f"The pipeline stopped during the "
                f"**{current['stage']}** stage: {str(exc)[:500]}"
            )

# ---------- Show an error, if there is one ----------
if "error" in st.session_state:
    st.error(st.session_state["error"])
    st.info("Check your API keys and internet connection, then try again.")

# ---------- Show results ----------
if "result" in st.session_state:
    result = st.session_state["result"]
    saved_topic = st.session_state.get("topic", "research-report")

    report = str(result.get("report", ""))
    feedback = str(result.get("feedback", ""))
    word_count = result.get("report_word_count") or count_words(report)

    if feedback.startswith("Critic review"):
        st.success("Research report generated successfully.")
    else:
        st.success("Research report generated and reviewed successfully.")

    # Summary row (stacks automatically on narrow screens)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Report word count", f"{word_count:,}")
    with col2:
        if 1000 <= word_count <= 1500:
            st.caption("Within the target range of about 1,000-1,500 words.")
        else:
            st.caption("Outside the target range of about 1,000-1,500 words.")

    # Download button
    st.download_button(
        label="⬇️ Download report (.md)",
        data=report,
        file_name=make_filename(saved_topic),
        mime="text/markdown",
    )

    # Report and critic feedback in tabs
    report_tab, critic_tab = st.tabs(["📄 Research Report", "🧐 Critic Feedback"])

    with report_tab:
        st.markdown(report)

    with critic_tab:
        if feedback.startswith("Critic review"):
            st.warning("The critic review did not complete, but your report was kept.")
            st.write(feedback)
        else:
            st.markdown(feedback)