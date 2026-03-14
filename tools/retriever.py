"""
tools/retriever.py - ChromaDB retrieval tool for course documents.

This tool searches the local vector store of course notes/documents
and returns the most relevant chunks for a given query.
Maps to Paper 1 (Agentic RAG) Section 4.2: Retriever Agent.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


def get_collection():
    """Get the ChromaDB collection with the embedding function."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(
        name="course_docs",
        embedding_function=ef
    )
    return collection


def search_course_docs(query: str, n_results: int = 3) -> str:
    """
    Search course documents in the ChromaDB vector store.

    Args:
        query: The search query about course material
        n_results: Number of results to return (default: 3)

    Returns:
        Formatted string with relevant document chunks and their sources
    """
    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results["documents"][0]:
            return "No relevant course documents found for this query."

        output = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            source = metadata.get("source", "Unknown")
            output.append(f"[Source: {source}]\n{doc}")

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return f"Error searching course documents: {str(e)}"


# Quick test
if __name__ == "__main__":
    print("Testing retriever...")
    result = search_course_docs("What is a star schema?")
    print(result)