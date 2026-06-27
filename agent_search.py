from ddgs import DDGS


QUERY = "Điều luật nào nói về Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"  # sửa query ở đây
NUM_CANDIDATES = 5


def web_search(query: str, count: int = 5) -> list[dict]:
    results = []

    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=count):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )

    return results


def main():
    print(f"Query: {QUERY}")
    print(f"Top {NUM_CANDIDATES} candidates")
    print("=" * 80)

    candidates = web_search(QUERY, count=NUM_CANDIDATES)

    if not candidates:
        print("Không tìm thấy kết quả.")
        return

    for i, item in enumerate(candidates, start=1):
        print(f"\nCandidate {i}")
        print(f"Title   : {item['title']}")
        print(f"URL     : {item['url']}")
        print(f"Snippet : {item['snippet']}")
        print("-" * 80)


if __name__ == "__main__":
    main()