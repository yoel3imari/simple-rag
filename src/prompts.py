"""Shared prompt templates for the RAG generation pipeline."""


def build_generation_prompt(question: str, context: str) -> str:
    """Build the LLM prompt from retrieved context and user question.

    Args:
        question: The user's original question.
        context: Retrieved context string (formatted chunks).

    Returns:
        A prompt string suitable for passing to ollama_chat().
    """
    return (
        "You are a helpful assistant. Use the context below to answer the question.\n"
        "\n"
        f"Context:\n{context}\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Answer:"
    )
