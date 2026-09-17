from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import insightmesh.graphs.research.analysts.nodes as analyst_nodes
import insightmesh.graphs.interview.nodes as interview_nodes
import insightmesh.graphs.research.nodes as research_nodes
from insightmesh.graphs.research.graph import builder
from insightmesh.schemas import Analyst, Perspectives, SearchQuery


class FakeLLM:
    def invoke(self, messages):
        instructions = messages[0].content
        request = messages[-1].content
        if "You are an analyst" in instructions:
            return AIMessage(content="What matters most?")
        if "You are an expert being interviewed" in instructions:
            return AIMessage(content="This matters [1].")
        if "technical writer creating a report" in instructions:
            return AIMessage(content="## Insights\nCombined body [1].\n\n## Sources\n[1] https://example.com/insights")
        if "Write the report introduction" in request:
            return AIMessage(content="# Title\n\n## Introduction\nIntro.")
        if "Write the report conclusion" in request:
            return AIMessage(content="## Conclusion\nEnd.")
        return AIMessage(content="## Section\n### Summary\nText [1]\n### Sources\n[1] https://example.com/insights")


class FakeStructuredLLM:
    def __init__(self, schema, calls):
        self.schema = schema
        self.calls = calls

    def invoke(self, messages):
        self.calls.append(self.schema)
        if self.schema is Perspectives:
            # One more than requested, to check the cap
            return Perspectives(analysts=[
                Analyst(affiliation="Lab", name=f"Analyst {i}", role="Researcher", description=f"Focus {i}")
                for i in range(3)
            ])
        return SearchQuery(search_query="what matters most")


def test_research_graph_end_to_end(monkeypatch):
    structured_calls, searches = [], []

    monkeypatch.setattr(interview_nodes, "get_llm", FakeLLM)
    monkeypatch.setattr(research_nodes, "get_llm", FakeLLM)
    for module in (analyst_nodes, interview_nodes):
        monkeypatch.setattr(module, "get_structured_llm", lambda schema: FakeStructuredLLM(schema, structured_calls))
    monkeypatch.setattr(interview_nodes, "search_duckduckgo", lambda q: searches.append(("ddg", q)) or '<Document href="https://example.com/insights"/>\nweb\n</Document>')
    monkeypatch.setattr(interview_nodes, "search_wikipedia", lambda q: searches.append(("wiki", q)) or "")

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}

    graph.invoke({"topic": "Testing", "max_analysts": 2, "max_num_turns": 1}, config)
    assert graph.get_state(config).next == ("human_feedback",)

    # Feedback regenerates, approval continues
    graph.invoke(Command(resume="add a skeptic"), config)
    assert structured_calls.count(Perspectives) == 2
    result = graph.invoke(Command(resume="perfect"), config)

    assert len(result["analysts"]) == 2
    assert len(result["sections"]) == 2
    # One query per interview turn, shared by both searches
    assert structured_calls.count(SearchQuery) == 2
    assert len(searches) == 4
    assert result["final_report"].startswith("# Title")
    assert result["final_report"].endswith("[1] https://example.com/insights")
