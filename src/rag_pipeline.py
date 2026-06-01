from src.vector_store import search_similar_chunks
from src.llm import MedicalChatbot

class RAGChatbot:
    def __init__(self):
        self.chatbot = MedicalChatbot()

    def chat(self, user_message, source_filter=None):
        # Step 1: Find relevant chunks
        relevant_chunks = search_similar_chunks(
            user_message,
            top_k=3,
            source_filter=source_filter
        )

        # Step 2: Build context
        context = "\n\n".join(relevant_chunks)

        # Step 3: Augment message with context
        augmented_message = f"""Use the following information to answer the question.

Context from medical documents:
{context}

User question: {user_message}

If the context doesn't contain relevant information, answer from your general knowledge.
"""

        # Step 4: Send to LLM
        return self.chatbot.chat(augmented_message)