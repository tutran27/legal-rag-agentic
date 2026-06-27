import json
import os
import re
import time
from typing import Any

import requests
import torch
from dotenv import load_dotenv

from src.common.config import settings


load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên nghiệp. Trả lời ngắn gọn, chính xác và "
    "đúng yêu cầu."
)
JSON_RETRY_ATTEMPTS = 2


def _log_retry(
    stage: str,
    reason: str,
    attempt: int,
    max_attempts: int,
    delay: float | None = None,
) -> None:
    delay_text = f" delay={delay:.1f}s" if delay is not None else ""
    print(
        f"[RETRY] stage={stage} reason={reason} "
        f"attempt={attempt}/{max_attempts}{delay_text}",
        flush=True,
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_env_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    values = []
    for part in re.split(r"[\n,;]+", raw):
        item = part.strip()
        if item:
            values.append(item)
    return values


def _load_openrouter_api_key(primary_key: str | None) -> str | None:
    return primary_key or settings.openrouter_api_key


def _load_groq_api_keys(primary_key: str | None) -> list[str]:
    keys = []
    for key in [primary_key, settings.groq_api_key]:
        if key and key not in keys:
            keys.append(key)

    for index in range(1, 13):
        key = os.getenv(f"GROQ_API_KEY_{index}")
        if key and key not in keys:
            keys.append(key)

    for key in _split_env_list(settings.groq_api_keys):
        if key not in keys:
            keys.append(key)
    return keys


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
    return (
        f"Câu hỏi gốc:\n{original_query}\n\n"
        f"Phản hồi trước không phải JSON hợp lệ:\n{bad_output[:4000]}\n\n"
        f"Yêu cầu: Chỉ trả lại một JSON object duy nhất theo đúng schema đã yêu cầu. "
        f"Không markdown, không giải thích, không thêm bất kỳ text nào khác ngoài JSON."
    )


class EndpointLLMClient:
    def __init__(
        self,
        endpoint_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint_url = (
            endpoint_url
            or settings.llm_endpoint_url
        ).rstrip("/")
        self.api_key = api_key or settings.qwen_api_key
        if not self.api_key:
            raise ValueError("Thiếu QWEN_API_KEY để dùng LLM_BACKEND=endpoint.")
        self.model = (
            model
            or settings.llm_endpoint_model
        )
        self.timeout = timeout or settings.llm_endpoint_timeout
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return {
            "backend": "endpoint",
            "model": self.model,
            "url": self.endpoint_url,
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
        token_limit = max_tokens or max_new_tokens or 1536
        response = self.session.post(
            f"{self.endpoint_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": token_limit,
                "temperature": temperature,
                "top_p": top_p,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                "Endpoint không trả về message content hợp lệ."
            ) from error
        if not isinstance(content, str):
            raise ValueError("Endpoint message content không phải chuỗi.")
        return content

    @staticmethod
    def extract_json_object(text: str) -> dict[str, Any]:
        """Trích xuất JSON object từ phản hồi LLM, kể cả khi có preamble."""
        text = text.strip()

        # 1) Thử parse nguyên văn — nhanh nhất nếu output đã là JSON thuần
        for strict in (True, False):
            try:
                return json.loads(text, strict=strict)
            except json.JSONDecodeError:
                pass

        # 2) Tìm vị trí dấu { đầu tiên và đếm ngoặc để lấy outermost JSON object
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for end in range(start, len(text)):
                ch = text[end]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(candidate, strict=False)
                        except json.JSONDecodeError:
                            break  # tiếp tục tìm nếu candidate không parse được

        raise ValueError(
            f"Không tìm thấy JSON object trong phản hồi LLM:\n{text}"
        )

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        retry_stage = str(kwargs.pop("retry_stage", "llm"))
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
                _log_retry(
                    retry_stage,
                    "invalid_json",
                    attempt + 1,
                    JSON_RETRY_ATTEMPTS,
                )
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "LLM trả về JSON không hợp lệ sau "
            f"{JSON_RETRY_ATTEMPTS + 1} lần thử:\n{output}"
        ) from last_error


class GroqLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        retry_attempts: int | None = None,
        retry_delay: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_keys = api_keys or _load_groq_api_keys(api_key)
        if not self.api_keys:
            raise ValueError("Thiếu GROQ_API_KEY để dùng LLM_BACKEND=groq.")
        self.api_key = self.api_keys[0]
        self.model = model or settings.groq_model
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.timeout = timeout or settings.groq_timeout
        self.retry_attempts = retry_attempts or settings.groq_retry_attempts
        self.retry_delay = retry_delay or float(settings.groq_retry_delay)
        self.session = session or requests.Session()
        self._next_api_key_index = 0

    def health(self) -> dict[str, Any]:
        return {
            "backend": "groq",
            "model": self.model,
            "api_key_count": len(self.api_keys),
        }

    def for_query(self) -> "GroqLLMClient":
        api_key = self.api_keys[self._next_api_key_index]
        self._next_api_key_index = (
            self._next_api_key_index + 1
        ) % len(self.api_keys)
        return GroqLLMClient(
            api_keys=[api_key],
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            retry_attempts=self.retry_attempts,
            retry_delay=self.retry_delay,
            session=self.session,
        )

    def generate(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        retry_stage: str = "llm",
        **_: Any,
    ) -> str:
        token_limit = max_tokens or max_new_tokens or 1536
        response = None
        for attempt in range(self.retry_attempts + 1):
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": token_limit,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=self.timeout,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt >= self.retry_attempts:
                break
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.replace(".", "", 1).isdigit()
                else self.retry_delay * (2**attempt)
            )
            _log_retry(
                retry_stage,
                f"http_{response.status_code}",
                attempt + 1,
                self.retry_attempts,
                delay,
            )
            time.sleep(delay)
        if response is None:
            raise RuntimeError("Không nhận được phản hồi từ Groq.")
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Groq không trả về message content hợp lệ.") from error
        if not isinstance(content, str):
            raise ValueError("Groq message content không phải chuỗi.")
        return content

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        retry_stage = str(kwargs.pop("retry_stage", "llm"))
        last_error: Exception | None = None
        output = ""
        current_query = query
        current_prompt = system_prompt
        for attempt in range(JSON_RETRY_ATTEMPTS + 1):
            output = self.generate(
                query=current_query,
                system_prompt=current_prompt,
                retry_stage=retry_stage,
                **kwargs,
            )
            try:
                return EndpointLLMClient.extract_json_object(output)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= JSON_RETRY_ATTEMPTS:
                    break
                _log_retry(
                    retry_stage,
                    "invalid_json",
                    attempt + 1,
                    JSON_RETRY_ATTEMPTS,
                )
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "Groq trả về JSON không hợp lệ sau "
            f"{JSON_RETRY_ATTEMPTS + 1} lần thử:\n{output}"
        ) from last_error


class OpenRouterLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        retry_attempts: int | None = None,
        retry_delay: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = _load_openrouter_api_key(api_key)
        if not self.api_key:
            raise ValueError(
                "Thiếu OPENROUTER_API_KEY để dùng LLM_BACKEND=openrouter."
            )
        self.model = (
            model
            or settings.openrouter_model
        )
        self.base_url = (
            base_url
            or settings.openrouter_base_url
        ).rstrip("/")
        self.timeout = timeout or settings.openrouter_timeout
        self.retry_attempts = (
            retry_attempts or settings.openrouter_retry_attempts
        )
        self.retry_delay = retry_delay or float(
            settings.openrouter_retry_delay
        )
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return {
            "backend": "openrouter",
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        retry_stage: str = "llm",
        **_: Any,
    ) -> str:
        token_limit = max_tokens or max_new_tokens or 1536
        response = None
        for attempt in range(self.retry_attempts + 1):
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://legal-rag-agent.local",
                    "X-Title": "Legal RAG Agent",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": token_limit,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=self.timeout,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt >= self.retry_attempts:
                break
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.replace(".", "", 1).isdigit()
                else self.retry_delay * (2**attempt)
            )
            _log_retry(
                retry_stage,
                f"http_{response.status_code}",
                attempt + 1,
                self.retry_attempts,
                delay,
            )
            time.sleep(delay)
        if response is None:
            raise RuntimeError("Không nhận được phản hồi từ OpenRouter.")
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                "OpenRouter không trả về message content hợp lệ."
            ) from error
        if not isinstance(content, str):
            raise ValueError("OpenRouter message content không phải chuỗi.")
        return content

    def call_llm_json(
        self,
        query: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        retry_stage = str(kwargs.pop("retry_stage", "llm"))
        last_error: Exception | None = None
        output = ""
        current_query = query
        current_prompt = system_prompt
        for attempt in range(JSON_RETRY_ATTEMPTS + 1):
            output = self.generate(
                query=current_query,
                system_prompt=current_prompt,
                retry_stage=retry_stage,
                **kwargs,
            )
            try:
                return EndpointLLMClient.extract_json_object(output)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= JSON_RETRY_ATTEMPTS:
                    break
                _log_retry(
                    retry_stage,
                    "invalid_json",
                    attempt + 1,
                    JSON_RETRY_ATTEMPTS,
                )
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "OpenRouter trả về JSON không hợp lệ sau "
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
            or settings.local_llm_model
        )
        self.max_model_len = (
            max_model_len
            if max_model_len is not None
            else (
                settings.local_llm_max_model_len
                if settings.local_llm_max_model_len > 0
                else None
            )
        )
        self.load_in_4bit = settings.local_llm_load_in_4bit
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
        retry_stage = str(kwargs.pop("retry_stage", "llm"))
        last_error: Exception | None = None
        output = ""
        current_query = query
        current_prompt = system_prompt
        for attempt in range(JSON_RETRY_ATTEMPTS + 1):
            output = self.generate(
                query=current_query,
                system_prompt=current_prompt,
                retry_stage=retry_stage,
                **kwargs,
            )
            try:
                return EndpointLLMClient.extract_json_object(output)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt >= JSON_RETRY_ATTEMPTS:
                    break
                _log_retry(
                    retry_stage,
                    "invalid_json",
                    attempt + 1,
                    JSON_RETRY_ATTEMPTS,
                )
                current_query = _json_repair_query(query, output)
                current_prompt = _json_repair_prompt(system_prompt)
        raise ValueError(
            "LLM trả về JSON không hợp lệ sau "
            f"{JSON_RETRY_ATTEMPTS + 1} lần thử:\n{output}"
        ) from last_error


def create_llm_client(
    backend: str | None = None,
    local_model: str | None = None,
) -> (
    EndpointLLMClient
    | GroqLLMClient
    | OpenRouterLLMClient
    | LocalQwenLLMClient
):
    selected = (backend or settings.llm_backend).lower()
    if selected == "groq":
        return GroqLLMClient()
    if selected == "openrouter":
        return OpenRouterLLMClient()
    if selected == "endpoint":
        return EndpointLLMClient()
    if selected == "local":
        return LocalQwenLLMClient(model_name=local_model)
    raise ValueError(
        "LLM_BACKEND chỉ hỗ trợ 'groq', 'openrouter', 'endpoint' hoặc 'local'."
    )


if __name__ == "__main__":
    client = create_llm_client()
    print(json.dumps(client.health(), ensure_ascii=False, indent=2))
    print(client.generate("Xin chào"))
