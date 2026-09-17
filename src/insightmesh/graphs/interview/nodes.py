import logging

from langchain.messages import SystemMessage, HumanMessage, AIMessage

from insightmesh.llm import get_llm, get_structured_llm
from insightmesh.schemas import Analyst, SearchQuery
from insightmesh.tools import search_duckduckgo, search_wikipedia
from insightmesh.graphs.interview.state import InterviewState
from insightmesh.graphs.interview.prompts import question_instructions, search_instructions, answer_instructions, section_writer_instructions

logger = logging.getLogger(__name__)

DEFAULT_MAX_NUM_TURNS = 2
END_OF_INTERVIEW = "Thank you so much for your help"


def _as_analyst(analyst) -> Analyst:
    """Checkpointing can hand the analyst back as a plain dict."""
    if isinstance(analyst, dict):
        return Analyst.model_validate(analyst)
    return analyst

def _speaker(message) -> str:
    """Analyst turns are unnamed AIMessages; the expert's are named, and the opening prompt is human."""
    if message.name == "expert" or not isinstance(message, AIMessage):
        return "expert"
    return "analyst"

def _as_speaker(messages, speaker: str):
    """Recast the interview so ``speaker``'s lines are AI turns and the rest human.

    Both the analyst and the expert are stored as AIMessages, so passing the raw
    history would end on an assistant turn, which models without prefill reject.
    """
    return [
        (AIMessage if _speaker(m) == speaker else HumanMessage)(content=m.text)
        for m in messages
    ]

def _transcript(messages) -> str:
    """Render the interview with real speaker labels (get_buffer_string calls both sides "AI")."""
    return "\n\n".join(f"{_speaker(m).capitalize()}: {m.text}" for m in messages)

def _format_context(context: list) -> str:
    return "\n\n---\n\n".join(context)


def generate_question(state: InterviewState):
    """ Node to generate a question """

    analyst = _as_analyst(state["analyst"])
    messages = state["messages"]

    system_message = question_instructions.format(goals=analyst.persona)
    question = get_llm().invoke([SystemMessage(content=system_message)]+_as_speaker(messages, "analyst"))

    return {"messages": [question]}

def generate_search_query(state: InterviewState):
    """ Turn the latest question into one query that every search node uses """

    structured_llm = get_structured_llm(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=search_instructions)]+[HumanMessage(content=_transcript(state["messages"]))])

    return {"search_query": search_query.search_query}

def search_web(state: InterviewState):
    """ Retrieve docs from web search (DuckDuckGo) """

    try:
        formatted_search_docs = search_duckduckgo(state["search_query"])
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return {"context": []}

    return {"context": [formatted_search_docs]} if formatted_search_docs else {"context": []}

def search_wiki(state: InterviewState):
    """ Retrieve docs from Wikipedia """

    try:
        formatted_search_docs = search_wikipedia(state["search_query"])
    except Exception:
        logger.exception("Wikipedia search failed")
        return {"context": []}

    return {"context": [formatted_search_docs]} if formatted_search_docs else {"context": []}

def generate_answer(state: InterviewState):
    """ Node to answer a question """

    # Get state
    analyst = _as_analyst(state["analyst"])
    messages = state["messages"]
    context = _format_context(state.get("context", []))

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = get_llm().invoke([SystemMessage(content=system_message)]+_as_speaker(messages, "expert"))

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """ Save interviews """

    return {"interview": _transcript(state["messages"])}

def write_section(state: InterviewState):
    """ Node to write a report section from the interview and its sources """
    
    interview = state["interview"]
    context = _format_context(state.get("context", []))
    analyst = _as_analyst(state["analyst"])

    system_message = section_writer_instructions.format(focus=analyst.description)
    section = get_llm().invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Interview transcript:\n\n{interview}\n\nSource documents:\n\n{context}"),
    ])

    return {"sections": [section.text]}


def route_messages(state: InterviewState, name: str = "expert"):
    """ Route between question and answer """

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', DEFAULT_MAX_NUM_TURNS)

    # Check the number of expert answers
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if END_OF_INTERVIEW in last_question.text:
        return 'save_interview'
    
    return "ask_question"
