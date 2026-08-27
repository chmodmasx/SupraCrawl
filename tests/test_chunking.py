from supracrawl.chunking import chunk_markdown, select_chunks


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
