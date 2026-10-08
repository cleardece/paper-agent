from agents.retriever import RetrieverAgent


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [[1.0] for _ in texts]


class FakeUserMemory:
    def __init__(self):
        self.interactions = []

    def get_interests(self, _user_id, top_k=5):
        return []

    def process_interaction(self, _user_id, query):
        self.interactions.append(query)


class FakeMongo:
    def __init__(self):
        self.user_memory = FakeUserMemory()

    def get_paper(self, paper_id):
        return {"arxiv_id": paper_id, "title": "Paper A"}


class RecordingRetriever(RetrieverAgent):
    def __init__(self, embedder, mongo):
        super().__init__(embedder, object(), mongo, llm=None)
        self.multi_query_input = None
        self.section_query = None

    def _multi_query(self, query, session_id=None, user_interests=None):
        self.multi_query_input = query
        return [query]

    def _detect_section_intent(self, query):
        self.section_query = query
        return None

    def _two_level_retrieval(self, query, query_vector, session_id=None,
                             paper_ids=None, graph_paper_ids=None):
        return [{
            "paper_arxiv_id": "P001",
            "chunk_index": 1,
            "content": "paper evidence",
            "score": 1.0,
            "section": "Experiment",
        }]


def test_retriever_uses_rewritten_query_but_records_original_user_message():
    embedder = FakeEmbedder()
    mongo = FakeMongo()
    retriever = RecordingRetriever(embedder, mongo)

    result = retriever.invoke({
        "user_query": "实验呢？",
        "retrieval_query": "Paper A 的实验设置、指标和结果是什么？",
        "user_id": "u1",
        "turn_context": {
            "primary_paper_id": "P001",
            "paper_ids": ["P001"],
        },
    })

    expected = "Paper A 的实验设置、指标和结果是什么？"
    assert retriever.multi_query_input == expected
    assert retriever.section_query == expected
    assert embedder.texts == [expected]
    assert mongo.user_memory.interactions == ["实验呢？"]
    assert result["retrieved_chunks"][0]["content"] == "paper evidence"
