import os
from typing import Any
import json
import re
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một trợ lý AI chuyên nghiệp, trả lời ngắn gọn, chính xác và đúng yêu cầu."
)

class GroqLLMClient:
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        query: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> str:
        used_max_tokens = max_tokens if max_tokens is not None else max_new_tokens
        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        completion = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=used_max_tokens,
        )
        content = completion.choices[0].message.content
        return content or ""

    def extract_json_object(self, text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Cannot find JSON object in LLM output:\n{text}")

        return json.loads(match.group(0))

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.extract_json_object(
            self.generate(query=query, system_prompt=system_prompt, **kwargs)
        )

if __name__=="__main__":
    model = GroqLLMClient()
    print(model.generate("Hello"))
