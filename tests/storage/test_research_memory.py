from copy import deepcopy

from storage.research_memory import ResearchMemoryService


class FakeResult:
    def __init__(self, inserted_id=None):
        self.inserted_id = inserted_id


class FakeCollection:
    def __init__(self):
        self.docs = []
        self.indexes = []

    def create_index(self, fields, **kwargs):
        self.indexes.append((fields, kwargs))

    def find_one(self, query):
        for doc in self.docs:
            if all(doc.get(key) == value for key, value in query.items()):
                return deepcopy(doc)
        return None

    def insert_one(self, doc):
        self.docs.append(deepcopy(doc))
        return FakeResult(len(self.docs))

    def update_one(self, query, update, upsert=False):
        for index, doc in enumerate(self.docs):
            if all(doc.get(key) == value for key, value in query.items()):
                doc.update(deepcopy(update.get("$set", {})))
                self.docs[index] = doc
                return FakeResult()
        if upsert:
            doc = deepcopy(query)
            doc.update(deepcopy(update.get("$setOnInsert", {})))
            doc.update(deepcopy(update.get("$set", {})))
            self.docs.append(doc)
        return FakeResult()

    def count_documents(self, query):
        return sum(
            all(doc.get(key) == value for key, value in query.items())
            for doc in self.docs
        )


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())


class FailingLLM:
    def invoke(self, _messages):
        raise RuntimeError("LLM unavailable")


def test_explicit_direction_correction_supersedes_prior_value():
    service = ResearchMemoryService(FakeDatabase(), llm=None)
    service.apply_updates(
        "local-user",
        [{"field": "research_directions", "value": ["computer vision"], "mode": "set", "confidence": 0.95}],
        "s1",
        ["m1"],
    )

    profile = service.apply_updates(
        "local-user",
        [{"field": "research_directions", "value": ["topology optimization"], "mode": "set", "confidence": 1.0, "explicit": True}],
        "s2",
        ["m2"],
    )

    assert profile["research_directions"] == ["topology optimization"]
    assert service.events.count_documents({"user_id": "local-user"}) == 2


def test_non_explicit_update_adds_values_without_erasing_prior_profile():
    service = ResearchMemoryService(FakeDatabase(), llm=None)
    service.apply_updates(
        "local-user",
        [{"field": "learning_status", "value": ["硕士研究生"], "confidence": 0.9}],
        "s1",
        ["m1"],
    )

    profile = service.apply_updates(
        "local-user",
        [{"field": "learning_status", "value": ["研一", "硕士研究生"], "confidence": 0.7}],
        "s2",
        ["m2"],
    )

    assert profile["learning_status"] == ["硕士研究生", "研一"]


def test_extraction_failure_returns_no_updates():
    service = ResearchMemoryService(FakeDatabase(), llm=FailingLLM())

    assert service.extract_updates("我正在复现实验", "", {}) == []
