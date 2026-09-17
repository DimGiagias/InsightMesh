from typing_extensions import Annotated, NotRequired, TypedDict
from typing import List, Optional
import operator
from insightmesh.schemas import Analyst


class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    max_num_turns: NotRequired[int]
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: List[Analyst]
    sections: Annotated[List[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str
