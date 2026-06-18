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
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
JSON_RETRY_ATTEMPTS = 2


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _json_repair_prompt(system_prompt: str | None) -> str:
    return "\n\n".join(
        [
            system_prompt or DEFAULT_SYSTEM_PROMPT,
            (
                "Phản hồi trước không phải JSON hợp lệ. "
                "Chỉ trả lại một JSON object hợp lệ, không markdown, "
                "không giải thích, không thêm chữ ngoài JSON."
            ),
        ]
    )


def _json_repair_query(original_query: str, bad_output: str) -> str:
    return json.dumps(
        {
            "original_input": original_query,
            "invalid_output": bad_output[:4000],
            "instruction": "Sửa thành JSON object hợp lệ theo đúng schema đã yêu cầu.",
        },
        ensure_ascii=False,
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
        last_error: Exception | None = None
        output = ""
        current_query = query
        current_prompt = system_prompt
        for attempt in range(JSON_RETRY_ATTEMPTS + 1):
            output = self.generate(
                query=current_query,
                system_prompt=current_prompt,
                **kwargs,
            )
            try:
                return self.extract_json_object(output)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= JSON_RETRY_ATTEMPTS:
                    break
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "LLM trả về JSON không hợp lệ sau "
            f"{JSON_RETRY_ATTEMPTS + 1} lần thử:\n{output}"
        ) from last_error


class LocalQwenLLMClient:
    def __init__(
        self,
        model_name: str | None = None,
        max_model_len: int | None = None
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = (
            model_name
            or os.getenv("LOCAL_LLM_MODEL")
            or DEFAULT_LOCAL_MODEL
        )
        env_max_model_len = os.getenv("LOCAL_LLM_MAX_MODEL_LEN")
        self.max_model_len = (
            max_model_len
            if max_model_len is not None
            else int(env_max_model_len) if env_max_model_len else None
        )
        self.load_in_4bit = _env_bool("LOCAL_LLM_LOAD_IN_4BIT", True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.model.eval()

    def health(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "model": self.model_name,
            "load_in_4bit": self.load_in_4bit,
        }

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
        import torch

        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenize_kwargs = {
            "return_tensors": "pt",
            "truncation": self.max_model_len is not None,
        }
        if self.max_model_len is not None:
            tokenize_kwargs["max_length"] = self.max_model_len
        inputs = self.tokenizer(text, **tokenize_kwargs).to(self.model.device)

        token_limit = max_tokens or max_new_tokens or 1536
        generation_kwargs = {
            "max_new_tokens": token_limit,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        output = ""
        current_query = query
        current_prompt = system_prompt
        for attempt in range(JSON_RETRY_ATTEMPTS + 1):
            output = self.generate(
                query=current_query,
                system_prompt=current_prompt,
                **kwargs,
            )
            try:
                return EndpointLLMClient.extract_json_object(output)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= JSON_RETRY_ATTEMPTS:
                    break
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "LLM trả về JSON không hợp lệ sau "
            f"{JSON_RETRY_ATTEMPTS + 1} lần thử:\n{output}"
        ) from last_error


def create_llm_client(
    backend: str | None = None,
    local_model: str | None = None,
) -> EndpointLLMClient | LocalQwenLLMClient:
    selected = (backend or os.getenv("LLM_BACKEND") or "endpoint").lower()
    if selected == "endpoint":
        return EndpointLLMClient()
    if selected == "local":
        return LocalQwenLLMClient(model_name=local_model)
    raise ValueError("LLM_BACKEND chỉ hỗ trợ 'endpoint' hoặc 'local'.")


if __name__ == "__main__":
    client = create_llm_client()
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))
    print(client.generate("Xin chào"))
