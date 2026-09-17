from typing_extensions import Annotated, NotRequired, TypedDict
from typing import List
import operator
from langgraph.graph import MessagesState
from insightmesh.schemas import Analyst


class InterviewState(MessagesState):
    max_num_turns: NotRequired[int]
    context: Annotated[List, operator.add]
    analyst: Analyst
    search_query: NotRequired[str]
    interview: NotRequired[str]
    sections: NotRequired[List[str]]


class InterviewOutputState(TypedDict):
    # Only this goes back to the parent, parallel interviews returning shared
    # keys like max_num_turns would otherwise collide.
    sections: List[str]
