"""Build the knowledge corpus (design section 8): download, extract and chunk.

`uv run python -m chhaaya.corpus` downloads every PDF in data/sources.csv to
data/pdfs/ (kept out of git), extracts text page by page, drops running headers,
footers, page numbers and contents-page lines, finds section headings by font,
and writes data/chunks.jsonl: chunks of MIN_TOKENS to MAX_TOKENS BGE-M3 tokens
that never cross a section. A section shorter than MIN_TOKENS stays one smaller
chunk. Each chunk keeps the source's title, publisher, year and URL, and the
PDF page (1-based, as a viewer numbers it) where the chunk starts.
"""

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx2
import pymupdf
from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parents[2] / "data"
SOURCES = DATA / "sources.csv"
PDFS = DATA / "pdfs"
CHUNKS = DATA / "chunks.jsonl"

TOKENIZER = "BAAI/bge-m3"
MIN_TOKENS = 300
MAX_TOKENS = 500

# Lines in the top or bottom MARGIN of a page are header/footer candidates.
MARGIN = 0.1
# A margin line repeated (digits aside) on this share of pages is a running header.
REPEATED = 0.3
# Larger than body text by this factor makes a line a heading.
HEADING_SCALE = 1.15
HEADING_MAX_WORDS = 15

_PAGE_NUMBER = re.compile(
    r"^[\s\-–—|]*(page\s*)?([0-9]+|[ivxlc]+)(\s*(of|/)\s*[0-9]+)?[\s\-–—|]*$",
    re.IGNORECASE,
)
_CONTENTS_LINE = re.compile(r"(\.\s?){4,}\s*[0-9ivxlc]+\s*$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    publisher: str
    year: int
    url: str
    reason: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    title: str
    publisher: str
    year: int
    url: str
    page: int
    section: str
    text: str
    tokens: int


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool
    top: float  # share of page height
    bottom: float


@dataclass
class _Section:
    heading: str
    page: int
    paragraphs: list[tuple[int, str]]  # (page, text)


class DownloadError(RuntimeError):
    pass


def read_sources(path: Path = SOURCES) -> list[Source]:
    sources = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, strict=True)
        for row in reader:
            # An unquoted comma (say, inside a URL) shifts every later field.
            if None in row or None in row.values():
                raise ValueError(
                    f"{path}: line {reader.line_num} has the wrong number of "
                    "fields; quote any field that contains a comma"
                )
            sources.append(
                Source(
                    id=row["id"],
                    title=row["title"],
                    publisher=row["publisher"],
                    year=int(row["year"]),
                    url=row["url"],
                    reason=row["reason"],
                )
            )
    return sources


def download(sources: Iterable[Source], directory: Path, http: httpx2.Client) -> None:
    """Fetch each source not already in `directory`; raise naming every failure."""
    directory.mkdir(parents=True, exist_ok=True)
    failures = []
    for source in sources:
        target = directory / f"{source.id}.pdf"
        if target.exists():
            continue
        try:
            response = http.get(source.url)
            response.raise_for_status()
        except httpx2.HTTPError as error:
            failures.append(f"{source.id}: {error}")
            continue
        if not response.content.startswith(b"%PDF"):
            kind = response.headers.get("content-type", "unknown type")
            failures.append(f"{source.id}: {source.url} did not return a PDF ({kind})")
            continue
        partial = target.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(target)
    if failures:
        raise DownloadError("could not download:\n  " + "\n  ".join(failures))


def chunk_source(source: Source, pdf: Path, count_tokens: TokenCounter) -> list[Chunk]:
    chunks = []
    for section in _sections(_strip_furniture(_read_lines(pdf))):
        for page, text in _pack(section.paragraphs, count_tokens):
            chunks.append(
                Chunk(
                    id=f"{source.id}:{len(chunks) + 1}",
                    source_id=source.id,
                    title=source.title,
                    publisher=source.publisher,
                    year=source.year,
                    url=source.url,
                    page=page,
                    section=section.heading,
                    text=text,
                    tokens=count_tokens(text),
                )
            )
    return chunks


def _read_lines(pdf: Path) -> list[list[list[_Line]]]:
    """Pages, each a list of text blocks, each a list of lines."""
    pages = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            height = page.rect.height
            blocks = []
            for block in page.get_text("dict", sort=True)["blocks"]:
                lines = []
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    main = max(spans, key=lambda s: len(s["text"]))
                    lines.append(
                        _Line(
                            text=" ".join("".join(s["text"] for s in spans).split()),
                            size=round(main["size"], 1),
                            bold=all(
                                s["flags"] & 16 or "bold" in s["font"].lower()
                                for s in spans
                            ),
                            top=line["bbox"][1] / height,
                            bottom=line["bbox"][3] / height,
                        )
                    )
                if lines:
                    blocks.append(lines)
            pages.append(blocks)
    return pages


def _margin_key(line: _Line) -> str | None:
    if line.top > MARGIN and line.bottom < 1 - MARGIN:
        return None
    return re.sub(r"[0-9]+", "#", line.text.lower())


def _strip_furniture(pages: list[list[list[_Line]]]) -> list[list[list[_Line]]]:
    seen = Counter(
        key
        for blocks in pages
        for key in {_margin_key(line) for block in blocks for line in block}
        if key
    )
    threshold = max(3, REPEATED * len(pages))
    repeated = {key for key, count in seen.items() if count >= threshold}

    def keep(line: _Line) -> bool:
        if _CONTENTS_LINE.search(line.text):
            return False
        key = _margin_key(line)
        return key is None or not (key in repeated or _PAGE_NUMBER.match(line.text))

    return [
        [kept for block in blocks if (kept := [line for line in block if keep(line)])]
        for blocks in pages
    ]


def _sections(pages: list[list[list[_Line]]]) -> list[_Section]:
    sizes = Counter()
    for blocks in pages:
        for block in blocks:
            for line in block:
                sizes[line.size] += len(line.text)
    if not sizes:
        return []
    body = sizes.most_common(1)[0][0]

    def heading(block: list[_Line]) -> bool:
        text = " ".join(line.text for line in block)
        if len(text.split()) > HEADING_MAX_WORDS or not re.search(r"[^\W\d_]", text):
            return False
        larger = all(line.size >= body * HEADING_SCALE for line in block)
        bold_alone = len(block) <= 2 and all(line.bold for line in block)
        return larger or bold_alone

    sections = [_Section(heading="", page=1, paragraphs=[])]
    for number, blocks in enumerate(pages, start=1):
        for block in blocks:
            if heading(block):
                text = " ".join(line.text for line in block)
                current = sections[-1]
                if not current.paragraphs and current.page == number:
                    # A heading straight under another, like a chapter title
                    # and its first section: keep both.
                    current.heading = f"{current.heading} / {text}".strip(" /")
                else:
                    sections.append(_Section(heading=text, page=number, paragraphs=[]))
            else:
                sections[-1].paragraphs.append((number, _join(block)))
    return [s for s in sections if s.paragraphs]


def _join(block: list[_Line]) -> str:
    text = ""
    for line in block:
        if text.endswith("-") and line.text[:1].islower():
            text = text[:-1] + line.text  # re-join a word hyphenated at line end
        else:
            text = f"{text} {line.text}" if text else line.text
    return text


def _units(
    paragraphs: list[tuple[int, str]], count_tokens: TokenCounter
) -> list[tuple[int, str, int]]:
    """Paragraphs, split into sentences and then word runs until each fits."""
    units = []
    for page, paragraph in paragraphs:
        pieces = [paragraph]
        if count_tokens(paragraph) > MAX_TOKENS:
            pieces = _SENTENCE_END.split(paragraph)
        for piece in pieces:
            tokens = count_tokens(piece)
            if tokens <= MAX_TOKENS:
                units.append((page, piece, tokens))
                continue
            words = piece.split()
            step = max(1, len(words) * MAX_TOKENS // (tokens + 1))
            for start in range(0, len(words), step):
                run = " ".join(words[start : start + step])
                units.append((page, run, count_tokens(run)))
    return units


def _pack(
    paragraphs: list[tuple[int, str]], count_tokens: TokenCounter
) -> list[tuple[int, str]]:
    """Group a section's text into even chunks, none over MAX_TOKENS.

    Each chunk aims at the size that splits what is left of the section evenly,
    so a section does not end in a scrap.
    """
    units = _units(paragraphs, count_tokens)
    chunks = []
    i = 0
    while i < len(units):
        remaining = sum(tokens for _, _, tokens in units[i:])
        target = remaining / math.ceil(remaining / MAX_TOKENS)
        page, text, size = units[i]
        parts = [text]
        i += 1
        while i < len(units):
            grown = size + units[i][2]
            if grown > MAX_TOKENS or grown - target > target - size:
                break
            parts.append(units[i][1])
            size = grown
            i += 1
        chunks.append((page, " ".join(parts)))
    return chunks


def _bge_m3_counter() -> TokenCounter:
    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def main() -> None:
    sources = read_sources()
    with httpx2.Client(
        follow_redirects=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0 (Chhaaya corpus builder)"},
    ) as http:
        download(sources, PDFS, http)

    count_tokens = _bge_m3_counter()
    total = 0
    with CHUNKS.open("w", encoding="utf-8") as out:
        for source in sources:
            chunks = chunk_source(source, PDFS / f"{source.id}.pdf", count_tokens)
            if not chunks:
                raise SystemExit(f"{source.id}: no text extracted (scanned PDF?)")
            for chunk in chunks:
                out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            sizes = [c.tokens for c in chunks]
            print(
                f"{source.id:28} {len(chunks):4} chunks, tokens min {min(sizes)} "
                f"median {statistics.median(sizes):.0f} max {max(sizes)}"
            )
            total += len(chunks)
    print(f"{total} chunks from {len(sources)} sources -> {CHUNKS}", file=sys.stderr)


if __name__ == "__main__":
    main()
