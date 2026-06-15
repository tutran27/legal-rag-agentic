from src.generation.llm_service import GroqLLMClient


def test_extract_json_accepts_raw_newline_inside_answer():
    client = GroqLLMClient.__new__(GroqLLMClient)
    output = '{"answer": "Dòng thứ nhất.\nDòng thứ hai."}'

    result = client.extract_json_object(output)

    assert result["answer"] == "Dòng thứ nhất.\nDòng thứ hai."


def test_extract_json_accepts_json_inside_markdown():
    client = GroqLLMClient.__new__(GroqLLMClient)
    output = '```json\n{"answer": "Nội dung trả lời."}\n```'

    result = client.extract_json_object(output)

    assert result == {"answer": "Nội dung trả lời."}
