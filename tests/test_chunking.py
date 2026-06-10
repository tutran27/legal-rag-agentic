from src.chunking.build_retrieval_corpus import split_article_text


def test_split_article_text_keeps_short_article():
    text = "1. Nội dung ngắn. 2. Nội dung tiếp theo."
    assert split_article_text(text, max_chars=100) == [text]


def test_split_article_text_splits_long_article():
    text = "1. " + ("A" * 80) + " 2. " + ("B" * 80)
    chunks = split_article_text(text, max_chars=100)
    assert len(chunks) >= 2
    assert "A" in chunks[0]
    assert any("B" in chunk for chunk in chunks[1:])


def test_split_article_text_drops_tiny_trailing_fragment():
    text = ("A" * 100) + "\n" + "x"
    chunks = split_article_text(text, max_chars=100)
    assert chunks == ["A" * 100]
