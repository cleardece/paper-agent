"""Verify the actual OpenAI-compatible wire payload, without network calls."""

import asyncio
import json

import httpx
import pytest

import config


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []
    options = []
    clients = []
    async_clients = []
    original_client = httpx.Client
    original_async_client = httpx.AsyncClient
    original_chat_openai = config.ChatOpenAI

    def handle(request):
        requests.append({
            "body": json.loads(request.content),
            "path": request.url.path,
            "authorization": request.headers["authorization"],
        })
        return httpx.Response(200, json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    transport = httpx.MockTransport(handle)

    def client(*args, **kwargs):
        kwargs["transport"] = transport
        instance = original_client(*args, **kwargs)
        clients.append(instance)
        return instance

    def async_client(*args, **kwargs):
        kwargs["transport"] = transport
        instance = original_async_client(*args, **kwargs)
        async_clients.append(instance)
        return instance

    def chat_openai(**kwargs):
        options.append(dict(kwargs))
        # Remote/chat paths normally let the SDK create clients. Inject test
        # transports there too, retaining the original options for assertions.
        if "http_client" not in kwargs:
            kwargs["http_client"] = client(trust_env=False)
        if "http_async_client" not in kwargs:
            kwargs["http_async_client"] = async_client(trust_env=False)
        return original_chat_openai(**kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(config, "ChatOpenAI", chat_openai)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr(config, "LLM_MODEL", "ordinary-chat")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://chat.example/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "chat-key")
    monkeypatch.setattr(config, "GRAPH_LLM_MODEL", "qwen3-graph:8b")
    monkeypatch.setattr(config, "GRAPH_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setattr(config, "GRAPH_LLM_API_KEY", "ollama")
    monkeypatch.setattr(config, "GRAPH_LLM_MAX_OUTPUT_TOKENS", 2000)
    monkeypatch.setattr(config, "GRAPH_LLM_REQUEST_TIMEOUT_SECONDS", 240)
    yield requests, options
    for instance in clients:
        instance.close()
    for instance in async_clients:
        asyncio.run(instance.aclose())


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_ollama_graph_sends_effective_output_limit(monkeypatch, captured_requests, host):
    monkeypatch.setattr(config, "GRAPH_LLM_BASE_URL", f"http://{host}:11434/v1")
    requests, options = captured_requests
    config.get_graph_llm().invoke("test")

    body = requests[0]["body"]
    assert body["model"] == "qwen3-graph:8b"
    assert body["temperature"] == 0
    assert body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 2000
    assert "max_completion_tokens" not in body
    assert requests[0]["path"] == "/v1/chat/completions"
    assert requests[0]["authorization"] == "Bearer ollama"
    assert options[0]["timeout"] == 240
    assert options[0]["max_retries"] == 0
    assert options[0]["http_client"]._trust_env is False
    assert options[0]["http_async_client"]._trust_env is False


def test_async_graph_sends_the_same_limit(captured_requests):
    requests, _ = captured_requests
    asyncio.run(config.get_graph_llm().ainvoke("test"))
    assert requests[0]["body"]["max_tokens"] == 2000
    assert "max_completion_tokens" not in requests[0]["body"]


def test_sensenova_graph_retains_legacy_token_field(monkeypatch, captured_requests):
    monkeypatch.setattr(config, "GRAPH_LLM_BASE_URL", "https://token.sensenova.cn/v1")
    monkeypatch.setattr(config, "GRAPH_LLM_MODEL", "deepseek-v4-flash")
    requests, options = captured_requests
    config.get_graph_llm().invoke("test")
    assert requests[0]["body"]["max_tokens"] == 2000
    assert "max_completion_tokens" not in requests[0]["body"]
    assert "http_client" not in options[0]


def test_other_graph_provider_retains_standard_token_field(monkeypatch, captured_requests):
    monkeypatch.setattr(config, "GRAPH_LLM_BASE_URL", "https://graph.example/v1")
    requests, _ = captured_requests
    config.get_graph_llm().invoke("test")
    assert requests[0]["body"]["max_completion_tokens"] == 2000
    assert "max_tokens" not in requests[0]["body"]


def test_ordinary_chat_keeps_its_own_model_and_behavior(captured_requests):
    requests, options = captured_requests
    config.get_llm().invoke("test")
    body = requests[0]["body"]
    assert body["model"] == "ordinary-chat"
    assert body["temperature"] == 0.3
    assert requests[0]["authorization"] == "Bearer chat-key"
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
    assert options[0]["timeout"] == 30
    assert options[0]["max_retries"] == 2
    assert "http_client" not in options[0]
    assert "http_async_client" not in options[0]
