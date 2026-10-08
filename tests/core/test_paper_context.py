from core.paper_context import PaperContextResolver, PaperFocusState


def paper(paper_id: str, title: str, doi: str | None = None) -> dict:
    return {
        "arxiv_id": paper_id,
        "title": title,
        "doi": doi,
        "abstract": "",
    }


class FakePaperRepository:
    def __init__(self, papers: list[dict]):
        self.papers = papers

    def get_paper(self, paper_id: str):
        return next(
            (item for item in self.papers if item["arxiv_id"] == paper_id),
            None,
        )

    def list_papers(self, **_kwargs):
        return list(self.papers)


def resolve(resolver, query, primary=None, explicit=None, history=None):
    return resolver.resolve(
        query=query,
        explicit_target_paper_id=explicit,
        paper_focus=PaperFocusState(
            primary_paper_id=primary,
            active_paper_ids=[primary] if primary else [],
        ),
        recent_paper_contexts=history or [],
    )


def test_session_primary_is_inherited_without_pronouns():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "Paper One")])
    )

    context = resolve(resolver, "总结实验结果", primary="P001")

    assert context.primary_paper_id == "P001"
    assert context.paper_ids == ["P001"]
    assert context.source == "session_focus"
    assert context.inherited is True


def test_explicit_target_switches_primary():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "One"), paper("P002", "Two")])
    )

    context = resolve(
        resolver,
        "分析新论文",
        primary="P001",
        explicit="P002",
    )

    assert context.status == "switch"
    assert context.primary_paper_id == "P002"
    assert context.paper_ids == ["P002"]
    assert context.source == "explicit_target"
    assert context.switched_from_paper_id == "P001"


def test_arxiv_id_in_message_has_priority_over_session_focus():
    resolver = PaperContextResolver(
        FakePaperRepository(
            [paper("P001", "One"), paper("2401.01234", "New Paper")]
        )
    )

    context = resolve(
        resolver,
        "分析 arXiv:2401.01234v2 的方法",
        primary="P001",
    )

    assert context.status == "switch"
    assert context.primary_paper_id == "2401.01234"
    assert context.source == "arxiv_id"


def test_doi_in_message_matches_local_paper():
    resolver = PaperContextResolver(
        FakePaperRepository(
            [paper("P001", "One", doi="10.1000/example.42")]
        )
    )

    context = resolve(resolver, "分析 DOI:10.1000/example.42")

    assert context.primary_paper_id == "P001"
    assert context.source == "doi"


def test_compare_keeps_current_primary_and_adds_title_match():
    resolver = PaperContextResolver(
        FakePaperRepository(
            [paper("P001", "Paper One"), paper("P002", "Paper B")]
        )
    )

    context = resolve(
        resolver,
        "把当前论文和 Paper B 对比一下",
        primary="P001",
    )

    assert context.primary_paper_id == "P001"
    assert context.paper_ids == ["P001", "P002"]
    assert context.status == "resolved"


def test_history_metadata_is_used_when_session_has_no_focus():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "Paper One")])
    )

    context = resolve(
        resolver,
        "总结实验结果",
        history=[{
            "primary_paper_id": "P001",
            "paper_ids": ["P001"],
            "status": "resolved",
            "source": "title_match",
            "confidence": 0.95,
            "inherited": False,
        }],
    )

    assert context.primary_paper_id == "P001"
    assert context.source == "history_resolution"
    assert context.inherited is True


def test_explicit_unscoped_search_does_not_become_analysis_target():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "Paper One")])
    )

    context = resolve(resolver, "帮我找 PINN 相关论文", primary="P001")

    assert context.primary_paper_id is None
    assert context.paper_ids == []
    assert context.status == "unresolved"


def test_similar_paper_search_retains_current_paper_as_search_context():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "Physics Informed Networks")])
    )

    context = resolve(resolver, "帮我找几篇类似论文", primary="P001")

    assert context.primary_paper_id == "P001"
    assert context.paper_ids == ["P001"]
    assert context.inherited is True


def test_multiple_title_matches_are_ambiguous_without_free_form_guessing():
    resolver = PaperContextResolver(
        FakePaperRepository(
            [
                paper("P001", "Graph Retrieval for Science"),
                paper("P002", "Graph Retrieval for Science Extended"),
            ]
        )
    )

    context = resolve(resolver, "分析 Graph Retrieval for Science Extended")

    assert context.status == "ambiguous"
    assert context.primary_paper_id is None
    assert context.paper_ids == ["P001", "P002"]


def test_invoke_reads_legacy_active_list_and_writes_serialized_context():
    resolver = PaperContextResolver(
        FakePaperRepository([paper("P001", "Paper One")])
    )

    result = resolver.invoke({
        "user_query": "总结实验结果",
        "target_paper_id": None,
        "active_paper_ids": ["P001"],
        "recent_paper_contexts": [],
    })

    assert result["paper_context"]["primary_paper_id"] == "P001"
    assert result["primary_paper_id"] == "P001"
