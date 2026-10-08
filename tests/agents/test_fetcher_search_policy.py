from agents.fetcher import FetcherAgent


class RecordingArxiv:
    def __init__(self, papers):
        self.papers = papers
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        return self.papers[:max_results]


class ExistingPaperMongo:
    def paper_exists(self, _paper_id):
        return True


def paper(paper_id, title):
    return {
        "arxiv_id": paper_id,
        "title": title,
        "abstract": "",
        "authors": [],
        "pdf_url": "https://example.test/paper.pdf",
    }


def state(mode, value, allowed=True):
    return {
        "user_query": "raw user analysis requirements",
        "session_id": None,
        "turn_context": {
            "query": "raw user analysis requirements",
            "intent": "search" if allowed else "analyze",
            "primary_paper_id": None,
            "paper_ids": [],
            "allow_external_search": allowed,
            "search_request": {"mode": mode, "value": value},
        },
    }


def make_agent(arxiv):
    return FetcherAgent(arxiv, None, ExistingPaperMongo(), None, None)


def test_id_search_uses_validated_request_and_returns_primary_id():
    arxiv = RecordingArxiv([paper("P003", "Paper Three")])

    result = make_agent(arxiv).invoke(state("arxiv_id", "P003"))

    assert arxiv.calls == [("P003", 1)]
    assert result["primary_paper_id"] == "P003"
    assert result["resolved_paper_ids"] == ["P003"]


def test_keyword_search_returns_all_ids_without_arbitrary_primary():
    arxiv = RecordingArxiv([
        paper("P003", "Paper Three"),
        paper("P004", "Paper Four"),
    ])

    result = make_agent(arxiv).invoke(state("keywords", "PINN fluids"))

    assert arxiv.calls == [("PINN fluids", 5)]
    assert result["primary_paper_id"] is None
    assert result["resolved_paper_ids"] == ["P003", "P004"]


def test_denied_search_returns_business_error_and_never_calls_arxiv():
    arxiv = RecordingArxiv([paper("P003", "Paper Three")])

    result = make_agent(arxiv).invoke(
        state("keywords", "raw user analysis requirements", allowed=False)
    )

    assert result["error"] == "EXTERNAL_SEARCH_NOT_ALLOWED"
    assert arxiv.calls == []
