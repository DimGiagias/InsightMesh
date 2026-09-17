from ddgs import DDGS
from langchain_community.document_loaders import WikipediaLoader


def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Retrieve docs from web search (DuckDuckGo)."""
    search_docs = DDGS().text(query, max_results=max_results)

    return "\n\n---\n\n".join(
        [
            f'<Document href="{doc["href"]}"/>\n{doc["title"]}\n{doc["body"]}\n</Document>'
            for doc in search_docs
        ]
    )


def search_wikipedia(query: str, max_docs: int = 3) -> str:
    """Retrieve docs from Wikipedia."""
    search_docs = WikipediaLoader(query=query, load_max_docs=max_docs).load()

    return "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
