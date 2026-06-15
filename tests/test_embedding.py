from src.common.embedding import HARRIER_QUERY_INSTRUCTION, embed_dense


class FakeModel:
    def __init__(self):
        self.texts = None

    def encode(self, texts, **kwargs):
        self.texts = texts
        return texts


def test_dense_query_uses_harrier_instruction():
    model = FakeModel()

    embed_dense(["Điều kiện hỗ trợ là gì?"], model, is_query=True)

    assert model.texts == [
        f"{HARRIER_QUERY_INSTRUCTION}Điều kiện hỗ trợ là gì?"
    ]


def test_dense_document_does_not_use_query_instruction():
    model = FakeModel()

    embed_dense(["Nội dung điều luật."], model)

    assert model.texts == ["Nội dung điều luật."]
