from src.rag_pipeline import RAGChatbot

bot = RAGChatbot()

response = bot.chat("what is this book about?")
print(response)