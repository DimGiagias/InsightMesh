import pytest
from langchain.messages import AIMessage, HumanMessage

from insightmesh.graphs.research.analysts.nodes import normalize_feedback
from insightmesh.graphs.interview.nodes import _as_speaker, _transcript, route_messages
from insightmesh.graphs.research.nodes import finalize_report, initiate_all_interviews, split_sources
from insightmesh.schemas import Analyst


def _interview(*turns):
    """Opening human prompt, then alternating analyst / expert turns."""
    messages = [HumanMessage(content="So you said you were writing an article on X?")]
    for i, text in enumerate(turns):
        messages.append(AIMessage(content=text, name="expert" if i % 2 else None))
    return messages


# finalize_report

def test_finalize_report_keeps_text_ending_in_insight_letters():
    state = {
        "introduction": "# Title\n\n## Introduction\nIntro.",
        "content": "## Insights\nBody.\n\n## Sources\n[1] https://example.com/insights",
        "conclusion": "## Conclusion\nEnd.",
    }
    report = finalize_report(state)["final_report"]

    assert report.endswith("[1] https://example.com/insights")
    assert "## Insights" not in report
    assert "Body." in report

def test_split_sources_tolerates_spacing_and_missing_section():
    assert split_sources("Body\n##  Sources  \n[1] a") == ("Body", "[1] a")
    assert split_sources("Body only") == ("Body only", None)



def test_route_messages_stops_after_max_turns():
    messages = _interview("Q1", "A1", "Q2", "A2")
    assert route_messages({"messages": messages}) == "save_interview"
    assert route_messages({"messages": messages, "max_num_turns": 3}) == "ask_question"

def test_route_messages_stops_when_analyst_says_thanks():
    messages = _interview("Thank you so much for your help!", "A1")
    assert route_messages({"messages": messages, "max_num_turns": 5}) == "save_interview"

def test_transcript_labels_both_speakers():
    transcript = _transcript(_interview("Q1", "A1"))
    assert transcript.splitlines()[0].startswith("Expert:")
    assert "Analyst: Q1" in transcript
    assert "Expert: A1" in transcript

def test_as_speaker_ends_on_human_turn_for_the_speaker():
    messages = _interview("Q1", "A1", "Q2")
    assert isinstance(_as_speaker(messages, "expert")[-1], HumanMessage)
    assert isinstance(_as_speaker(_interview("Q1", "A1"), "analyst")[-1], HumanMessage)



@pytest.mark.parametrize("feedback", [None, "", "  perfect ", "Continue", {"feedback": "yes"}, {"feedback": None}])
def test_normalize_feedback_approves(feedback):
    assert normalize_feedback(feedback) is None

def test_normalize_feedback_returns_text():
    assert normalize_feedback(" add an economist ") == "add an economist"
    assert normalize_feedback({"feedback": "add an economist"}) == "add an economist"

def test_normalize_feedback_rejects_unexpected_types():
    with pytest.raises(ValueError):
        normalize_feedback({"approved": True})


def test_initiate_all_interviews_passes_max_num_turns():
    analyst = Analyst(affiliation="a", name="n", role="r", description="d")
    base = {"topic": "T", "analysts": [analyst, analyst]}

    sends = initiate_all_interviews({**base, "max_num_turns": 4})
    assert len(sends) == 2
    assert all(s.arg["max_num_turns"] == 4 for s in sends)

    assert "max_num_turns" not in initiate_all_interviews(base)[0].arg
    assert initiate_all_interviews({**base, "human_analyst_feedback": "more"}) == "create_analysts"
