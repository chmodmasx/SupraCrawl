import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[\w.-]+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SEPARATOR = "\n\n---\n\n"


@dataclass(slots=True)
class Chunk:
    ordinal: int
    section_path: tuple[str, ...]
    text: str
    approx_tokens: int


def approx_tokens(text: str) -> int:
    """Conservative tokenizer-agnostic estimate for context budgeting.

    Latin-script content is estimated at roughly three characters/token.
    Non-Latin code points are charged at one token each to avoid severe
    underestimation for CJK and other scripts when the target LLM tokenizer is
    unknown.
    """
    if not text:
        return 0
    latin_chars = sum(1 for char in text if ord(char) <= 0x024F)
    non_latin_chars = len(text) - latin_chars
    return max(1, (latin_chars + 2) // 3 + non_latin_chars)


def _split_long_line(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]

    words = line.split()
    if len(words) <= 1:
        return [line[start : start + max_chars] for start in range(0, len(line), max_chars)]

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        added = len(word) + (1 if current else 0)
        if current and size + added > max_chars:
            pieces.append(" ".join(current))
            current = []
            size = 0
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                size = 0
            pieces.extend(
                word[start : start + max_chars] for start in range(0, len(word), max_chars)
            )
            continue
        current.append(word)
        size += len(word) + (1 if size else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_markdown(markdown: str, target_tokens: int = 500) -> list[Chunk]:
    target_chars = max(800, target_tokens * 3)
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

    for original_line in markdown.splitlines():
        heading = _HEADING_RE.match(original_line)
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
            buffer.append(original_line)
            size += len(original_line)
            continue

        for line in _split_long_line(original_line, target_chars):
            exceeds_target = size and size + len(line) + 1 > target_chars
            is_structured_line = line.lstrip().startswith(("- ", "* ", "|"))
            preserve_structure = is_structured_line and size < target_chars * 2
            if exceeds_target and not preserve_structure:
                flush()
            buffer.append(line)
            size += len(line) + 1

    flush()
    return chunks


def _query_terms(query: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(query) if len(token) > 1}


def _clip_text_to_tokens(text: str, max_tokens: int) -> str:
    if approx_tokens(text) <= max_tokens:
        return text

    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if approx_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    clipped = text[:low].rstrip()
    if low < len(text) and clipped:
        suffix = " …"
        while clipped and approx_tokens(clipped + suffix) > max_tokens:
            clipped = clipped[:-1].rstrip()
        clipped += suffix
    return clipped


def select_chunks(
    chunks: list[Chunk],
    query: str | None,
    max_tokens: int,
    max_chunks: int,
) -> list[Chunk]:
    if not chunks or max_tokens <= 0:
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
    separator_tokens = approx_tokens(_SEPARATOR)
    for chunk in ordered:
        if len(selected) >= max_chunks or budget >= max_tokens:
            break

        extra_separator = separator_tokens if selected else 0
        remaining = max_tokens - budget - extra_separator
        if remaining <= 0:
            break

        if chunk.approx_tokens <= remaining:
            selected.append(chunk)
            budget += extra_separator + chunk.approx_tokens
            continue

        clipped_text = _clip_text_to_tokens(chunk.text, remaining)
        if clipped_text:
            clipped_tokens = approx_tokens(clipped_text)
            selected.append(
                Chunk(
                    ordinal=chunk.ordinal,
                    section_path=chunk.section_path,
                    text=clipped_text,
                    approx_tokens=clipped_tokens,
                )
            )
            budget += extra_separator + clipped_tokens
        break

    return sorted(selected, key=lambda chunk: chunk.ordinal)
