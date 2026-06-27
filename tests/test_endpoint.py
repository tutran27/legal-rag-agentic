from unittest.mock import Mock

from src.generation.endpoint import (
    EndpointLLMClient,
    GroqLLMClient,
    create_llm_client,
)


def test_factory_uses_endpoint_by_default(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    client = create_llm_client()

    assert isinstance(client, EndpointLLMClient)


def test_generate_calls_endpoint_with_compatible_payload():
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "Nội dung trả lời"}}]
    }
    session = Mock()
    session.post.return_value = response
    client = EndpointLLMClient(
        endpoint_url="https://example.test/",
        api_key="test-key",
        model="test-model",
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
    session.post.assert_called_once_with(
        "https://example.test/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "Chỉ trả lời JSON"},
                {"role": "user", "content": "Câu hỏi"},
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        timeout=30,
    )
    response.raise_for_status.assert_called_once()


def test_call_llm_json_accepts_markdown_response():
    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"answer": "Nội dung"}\n```'
                }
            }
        ]
    }
    session = Mock()
    session.post.return_value = response
    client = EndpointLLMClient(
        endpoint_url="https://example.test",
        api_key="test-key",
        session=session,
    )

    result = client.call_llm_json("Câu hỏi")

    assert result == {"answer": "Nội dung"}


def test_extract_json_accepts_raw_newline_inside_answer():
    output = '{"answer": "Dòng thứ nhất.\nDòng thứ hai."}'

    result = EndpointLLMClient.extract_json_object(output)

    assert result["answer"] == "Dòng thứ nhất.\nDòng thứ hai."


def test_call_llm_json_retries_invalid_json_response(capsys):
    first = Mock()
    first.json.return_value = {
        "choices": [
            {"message": {"content": '{"answer": "thiếu đóng json"'}}
        ]
    }
    second = Mock()
    second.json.return_value = {
        "choices": [
            {"message": {"content": '{"answer": "Nội dung hợp lệ"}'}}
        ]
    }
    session = Mock()
    session.post.side_effect = [first, second]
    client = EndpointLLMClient(
        endpoint_url="https://example.test",
        api_key="test-key",
        session=session,
    )

    result = client.call_llm_json("Câu hỏi", max_new_tokens=200)

    assert result == {"answer": "Nội dung hợp lệ"}
    assert session.post.call_count == 2
    assert "stage=llm reason=invalid_json attempt=1/2" in capsys.readouterr().out
    retry_payload = session.post.call_args_list[1].kwargs["json"]
    # Với format mới, retry query là plain text chứa câu hỏi gốc và output lỗi
    assert "Câu hỏi" in retry_payload["messages"][1]["content"]
    assert "thiếu đóng json" in retry_payload["messages"][1]["content"]
    assert "JSON object duy nhất" in retry_payload["messages"][1]["content"]
    assert "JSON object hợp lệ" in retry_payload["messages"][0]["content"]


def test_groq_generate_calls_chat_completions_payload():
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "Nội dung trả lời"}}]
    }
    session = Mock()
    session.post.return_value = response
    client = GroqLLMClient(
        api_key="test-key",
        model="test-model",
        base_url="https://groq.test/openai/v1",
        timeout=30,
        session=session,
    )

    result = client.generate(
        query="Câu hỏi",
        system_prompt="Chỉ trả lời JSON",
        max_new_tokens=128,
        temperature=0.1,
    )

    assert result == "Nội dung trả lời"
    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["json"]["model"] == "test-model"
    assert kwargs["json"]["max_tokens"] == 128
    assert kwargs["json"]["messages"][0]["content"] == "Chỉ trả lời JSON"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_groq_logs_http_retry_stage(capsys):
    limited = Mock(status_code=429, headers={"Retry-After": "0"})
    success = Mock(status_code=200, headers={})
    success.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    session = Mock()
    session.post.side_effect = [limited, success]
    client = GroqLLMClient(api_key="test-key", session=session)

    result = client.generate("query", retry_stage="reasoning")

    assert result == "ok"
    assert "stage=reasoning reason=http_429 attempt=1/4" in capsys.readouterr().out


def test_groq_rotates_api_keys_per_query():
    client = GroqLLMClient(
        api_keys=["key-1", "key-2", "key-3"],
        session=Mock(),
    )

    query_1 = client.for_query()
    query_2 = client.for_query()
    query_3 = client.for_query()
    query_4 = client.for_query()

    assert query_1.api_key == "key-1"
    assert query_2.api_key == "key-2"
    assert query_3.api_key == "key-3"
    assert query_4.api_key == "key-1"
    assert query_1.api_keys == ["key-1"]
    assert query_4.api_keys == ["key-1"]
