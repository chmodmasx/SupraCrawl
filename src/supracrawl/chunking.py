import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    section_path: tuple[str, ...]
    text: str
    approx_tokens: int


def approx_tokens(text: str) -> int:
    # Deliberately tokenizer-agnostic. Replace with the target model tokenizer
    # when Hermes exposes it to the provider.
    return max(1, (len(text) + 3) // 4)


def chunk_markdown(markdown: str, target_tokens: int = 500) -> list[Chunk]:
    target_chars = max(800, target_tokens * 4)
    sections: list[str] = []
    chunks: list[Chunk] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    section_path=tuple(sections),
                    text=text,
                    approx_tokens=approx_tokens(text),
                )
            )
        buffer = []
        size = 0

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            sections[:] = sections[: level - 1]
            while len(sections) < level - 1:
                sections.append("")
            if len(sections) == level - 1:
                sections.append(title)
            else:
                sections[level - 1] = title
            buffer.append(line)
            size += len(line)
            continue

        exceeds_target = size and size + len(line) + 1 > target_chars
        is_structured_line = line.lstrip().startswith(("- ", "* ", "|"))
        if exceeds_target and not is_structured_line:
            flush()
        buffer.append(line)
        size += len(line) + 1

    flush()
    return chunks


def _query_terms(query: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(query) if len(token) > 1}


def select_chunks(
    chunks: list[Chunk],
    query: str | None,
    max_tokens: int,
    max_chunks: int,
) -> list[Chunk]:
    if not chunks:
        return []

    if query:
        terms = _query_terms(query)

        def score(chunk: Chunk) -> tuple[float, int]:
            text = chunk.text.lower()
            heading = " ".join(chunk.section_path).lower()
            matches = sum(text.count(term) for term in terms)
            heading_matches = sum(1 for term in terms if term in heading)
            density = matches / max(1, len(_WORD_RE.findall(text)))
            return (heading_matches * 4.0 + matches + density * 100.0, -chunk.ordinal)

        ordered = sorted(chunks, key=score, reverse=True)
    else:
        ordered = chunks

    selected: list[Chunk] = []
    budget = 0
    for chunk in ordered:
        if len(selected) >= max_chunks:
            break
        if selected and budget + chunk.approx_tokens > max_tokens:
            continue
        selected.append(chunk)
        budget += chunk.approx_tokens
        if budget >= max_tokens:
            break

    return sorted(selected, key=lambda chunk: chunk.ordinal)
