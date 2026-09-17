import re
from typing import Literal

from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from insightmesh.llm import get_llm
from insightmesh.graphs.research.state import ResearchGraphState
from insightmesh.graphs.research.prompts import report_writer_instructions, intro_conclusion_instructions

SOURCES_HEADER = re.compile(r"^##\s*Sources\s*$", re.MULTILINE)


def _formatted_sections(state: ResearchGraphState) -> str:
    return "\n\n".join(state["sections"])

def _write_bookend(state: ResearchGraphState, part: Literal["introduction", "conclusion"]) -> str:
    """Write the introduction or conclusion; they share one prompt."""
    system_message = intro_conclusion_instructions.format(topic=state["topic"], formatted_str_sections=_formatted_sections(state))
    response = get_llm().invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write the report {part}")])
    return response.text

def split_sources(content: str) -> tuple[str, str | None]:
    """Split the report body from its trailing ``## Sources`` section, if there is one."""
    parts = SOURCES_HEADER.split(content, maxsplit=1)
    if len(parts) == 1:
        return content, None
    body, sources = parts
    return body.rstrip(), sources.strip()


def write_report(state: ResearchGraphState):
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=state["topic"], context=_formatted_sections(state))
    report = get_llm().invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Write a report based upon these memos.")])
    return {"content": report.text}

def write_introduction(state: ResearchGraphState):
    return {"introduction": _write_bookend(state, "introduction")}

def write_conclusion(state: ResearchGraphState):
    return {"conclusion": _write_bookend(state, "conclusion")}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    content = state["content"].strip().removeprefix("## Insights").strip()
    content, sources = split_sources(content)

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    human_analyst_feedback = state.get('human_analyst_feedback')
    
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    topic = state["topic"]
    interview_input = {"messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]}
    if "max_num_turns" in state:
        interview_input["max_num_turns"] = state["max_num_turns"]

    return [Send("conduct_interview", {"analyst": analyst, **interview_input}) for analyst in state["analysts"]]
