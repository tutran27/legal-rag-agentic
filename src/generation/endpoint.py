import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ENDPOINT_URL = (
    "https://dinhtu7cfc--qwen2-5-14b-instruct-api-api.modal.run"
)
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên nghiệp. Trả lời ngắn gọn, chính xác và "
    "đúng yêu cầu."
)


class EndpointLLMClient:
    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint_url = (
            endpoint_url
            or os.getenv("LLM_ENDPOINT_URL")
            or DEFAULT_ENDPOINT_URL
        ).rstrip("/")
        self.timeout = timeout or int(os.getenv("LLM_ENDPOINT_TIMEOUT", "600"))
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.endpoint_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        **_: Any,
    ) -> str:
        token_limit = max_tokens or max_new_tokens or 1536
        response = self.session.post(
            f"{self.endpoint_url}/generate",
            json={
                "query": query,
                "prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
                "max_new_tokens": token_limit,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response")
        if not isinstance(content, str):
            raise ValueError(
                "Endpoint không trả về trường 'response' dạng chuỗi."
            )
        return content

    @staticmethod
    def extract_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        for strict in (True, False):
            try:
                return json.loads(text, strict=strict)
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(
                f"Không tìm thấy JSON object trong phản hồi LLM:\n{text}"
            )

        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return json.loads(json_text, strict=False)

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        output = self.generate(
            query=query,
            system_prompt=system_prompt,
            **kwargs,
        )
        return self.extract_json_object(output)


if __name__ == "__main__":
    client = EndpointLLMClient()
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))
    print(client.generate("Xin chào"))
