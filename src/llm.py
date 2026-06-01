import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MedicalChatbot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.conversation_history = []
        self.system_prompt = """You are MediBot, a helpful medical assistant.

Rules you must always follow:
1. Never give a definitive diagnosis
2. Always recommend consulting a real doctor for serious concerns
3. Be empathetic and clear
4. If someone describes an emergency, tell them to call emergency services immediately
"""

    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
        )

        assistant_message = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message
