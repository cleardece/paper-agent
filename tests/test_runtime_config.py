from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_example_environment_has_no_secret_value():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LLM_API_KEY=" in content
    assert "sk-" not in content


def test_compose_uses_a_portable_data_root():
    content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${PA_DATA_ROOT:-./.local-data}" in content
    assert "E:\\milvus" not in content


def test_readme_explains_chunked_index_recovery():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "chunked" in content
    assert "indexed" in content
    assert "补索引" in content


def test_graph_environment_defines_durable_tpm_recovery_limits():
    content = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "GRAPH_LLM_TPM_LIMIT=8000" in content
    assert "GRAPH_LLM_MAX_OUTPUT_TOKENS=2000" in content
    assert "GRAPH_LLM_PROMPT_OVERHEAD_TOKENS=1000" in content
    assert "GRAPH_RATE_LIMIT_BASE_DELAY_SECONDS=300" in content
    assert "GRAPH_RATE_LIMIT_MAX_DELAY_SECONDS=3600" in content
