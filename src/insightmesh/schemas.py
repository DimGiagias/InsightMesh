from pydantic import BaseModel, Field
from typing import List


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns and motives")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nAffiliation: {self.affiliation}\nRole: {self.role}\nDescription: {self.description}"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their role and affiliation.")


class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for retrieval.")
