from typing_extensions import NotRequired, TypedDict
from typing import List, Optional
from insightmesh.schemas import Analyst


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: NotRequired[Optional[str]]
    analysts: NotRequired[List[Analyst]]
