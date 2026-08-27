from supracrawl.chunking import approx_tokens, chunk_markdown, select_chunks


def test_query_aware_selection_prefers_matching_section() -> None:
    markdown = """# Intro
General information about the project.

## Installation
Install with Docker Compose and configure the service.

## Security
SSRF protection blocks private IP ranges and validates redirects.
"""
    chunks = chunk_markdown(markdown, target_tokens=10)
    selected = select_chunks(chunks, query="SSRF private IP", max_tokens=1000, max_chunks=1)
    assert len(selected) == 1
    assert "SSRF" in selected[0].text


def test_long_paragraph_is_split_before_selection() -> None:
    markdown = "# Long\n" + "word " * 2000
    chunks = chunk_markdown(markdown, target_tokens=100)
    assert len(chunks) > 1
    assert max(chunk.approx_tokens for chunk in chunks) < 300


def test_selection_never_exceeds_context_budget() -> None:
    markdown = "# Huge\n" + ("important content " * 2000)
    chunks = chunk_markdown(markdown, target_tokens=500)
    selected = select_chunks(chunks, query="important", max_tokens=128, max_chunks=3)
    payload = "\n\n---\n\n".join(chunk.text for chunk in selected)
    assert approx_tokens(payload) <= 128


def test_non_latin_text_is_not_severely_undercounted() -> None:
    text = "検索結果の品質を改善する" * 20
    assert approx_tokens(text) >= len(text)
