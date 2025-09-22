from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI, OpenAI

from app.core.config import get_settings

settings = get_settings()


class OpenAIChat:

    def __init__(self, model: Optional[str] = settings.openai_model) -> None:
        self.api_key = settings.openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key) if (self.api_key) else None

    @staticmethod
    def build_messages(question: str, passages: list[str]):
        system_prompt = 'You answer questions strictly from the provided context. If not present, reply: "I don\'t know."'
        user_prompt = (
            "To provide some context, here are some passages that might be related:\n\n"
        )

        for passage in passages:
            user_prompt += f"{passage}\n\n"
        user_prompt += f"And now the question for you:\n\n{question}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Answer:"},
        ]

    async def __call__(self, question: str, passages: list[str]) -> str:
        if not self.client:
            return "I don't know."

        messages = self.build_messages(question, passages)
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return "I don't know."
