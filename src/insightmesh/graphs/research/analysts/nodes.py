from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

from insightmesh.llm import get_structured_llm
from insightmesh.schemas import Perspectives
from insightmesh.graphs.research.analysts.state import GenerateAnalystsState
from insightmesh.graphs.research.analysts.prompts import analyst_instructions

APPROVAL_WORDS = {"", "perfect", "continue", "approved", "approve", "yes", "ok"}


def normalize_feedback(feedback) -> str | None:
    """Return the feedback text to regenerate with, or None if the analysts are approved.

    Accepts a plain string or ``{"feedback": "..."}``, which is what Studio sends
    when resuming with JSON.
    """
    if feedback is None:
        return None
    if isinstance(feedback, dict) and "feedback" in feedback:
        feedback = feedback["feedback"]
        if feedback is None:
            return None
    if not isinstance(feedback, str):
        raise ValueError(
            f"Expected feedback as a string or {{'feedback': str}}, got {type(feedback).__name__}: {feedback!r}"
        )

    feedback = feedback.strip()
    if feedback.lower() in APPROVAL_WORDS:
        return None
    return feedback

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """

    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback') or ""

    structured_llm = get_structured_llm(Perspectives)

    system_message = analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts)

    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])

    # Models sometimes return more analysts than asked for
    return {"analysts": analysts.analysts[:max_analysts]}

def human_feedback(state: GenerateAnalystsState):
    feedback = interrupt({
        "question": "Are these analysts okay?",
        "analysts": [
            analyst.model_dump() if hasattr(analyst, "model_dump") else analyst
            for analyst in state.get("analysts", [])
        ],
        "instructions": "Return feedback (a string, or {\"feedback\": \"...\"}) to regenerate analysts, or return empty/perfect/continue to approve."
    })

    return {"human_analyst_feedback": normalize_feedback(feedback)}
