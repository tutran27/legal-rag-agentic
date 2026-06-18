from unittest.mock import Mock

from src.generation.endpoint import GroqLLMClient, create_llm_client


def test_groq_generate_calls_openai_compatible_api():
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "Nội dung trả lời"}}]
    }
    session = Mock()
    session.post.return_value = response
    client = GroqLLMClient(
        api_key="test-key",
        model="test-model",
        base_url="https://groq.test/openai/v1/",
        timeout=30,
        session=session,
    )

    result = client.generate(
        query="Câu hỏi",
        system_prompt="Chỉ trả lời JSON",
        max_new_tokens=256,
        temperature=0.1,
    )

    assert result == "Nội dung trả lời"
    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "test-model"
    assert kwargs["json"]["messages"][0]["content"] == "Chỉ trả lời JSON"
    assert kwargs["json"]["messages"][1]["content"] == "Câu hỏi"
    assert kwargs["json"]["max_tokens"] == 256
    response.raise_for_status.assert_called_once()


def test_create_llm_client_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    client = create_llm_client()

    assert isinstance(client, GroqLLMClient)
    assert client.model == "test-model"
