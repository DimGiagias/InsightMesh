from langgraph.graph import StateGraph, START, END

from insightmesh.graphs.interview.state import InterviewState, InterviewOutputState
from insightmesh.graphs.interview.nodes import generate_question, generate_search_query, search_web, search_wiki, generate_answer, save_interview, write_section, route_messages

interview_builder = StateGraph(InterviewState, output_schema=InterviewOutputState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("generate_search_query", generate_search_query)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wiki)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "generate_search_query")
interview_builder.add_edge("generate_search_query", "search_web")
interview_builder.add_edge("generate_search_query", "search_wikipedia")
interview_builder.add_edge(["search_web", "search_wikipedia"], "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

graph = interview_builder.compile()
