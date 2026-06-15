from unittest.mock import Mock

from src.generation.endpoint import EndpointLLMClient


def test_generate_calls_endpoint_with_compatible_payload():
    response = Mock()
    response.json.return_value = {"response": "Nội dung trả lời"}
    session = Mock()
    session.post.return_value = response
    client = EndpointLLMClient(
        endpoint_url="https://example.test/",
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
        "https://example.test/generate",
        json={
            "query": "Câu hỏi",
            "prompt": "Chỉ trả lời JSON",
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        timeout=30,
    )
    response.raise_for_status.assert_called_once()


def test_call_llm_json_accepts_markdown_response():
    response = Mock()
    response.json.return_value = {
        "response": '```json\n{"answer": "Nội dung"}\n```'
    }
    session = Mock()
    session.post.return_value = response
    client = EndpointLLMClient(
        endpoint_url="https://example.test",
        session=session,
    )

    result = client.call_llm_json("Câu hỏi")

    assert result == {"answer": "Nội dung"}


def test_extract_json_accepts_raw_newline_inside_answer():
    output = '{"answer": "Dòng thứ nhất.\nDòng thứ hai."}'

    result = EndpointLLMClient.extract_json_object(output)

    assert result["answer"] == "Dòng thứ nhất.\nDòng thứ hai."
