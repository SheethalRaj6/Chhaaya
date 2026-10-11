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
from itertools import groupby
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
# A section starts at a heading set at least this much larger than body text.
# Bold body-size lines ("Remember:", "Key messages") stay inside their section.
HEADING_SCALE = 1.15
HEADING_MAX_WORDS = 15
# Text this much smaller than body text is a figure or chart label: dropped.
LABEL_SCALE = 0.65
# A contents page sits in the first CONTENTS_SHARE of a document, and most of
# its lines end in a page number.
CONTENTS_SHARE = 0.1

_PAGE_NUMBER = re.compile(
    r"^[\s\-–—|]*(page\s*)?([0-9]+|[ivxlc]+)(\s*(of|/)\s*[0-9]+)?[\s\-–—|]*$",
    re.IGNORECASE,
)
_CONTENTS_LINE = re.compile(r"(\.\s?){4,}\s*[0-9ivxlc]+\s*$", re.IGNORECASE)
# A contents line: "Role of ASHA 7", or a bare "7" or "1." in a column layout.
_LISTED = re.compile(r"(.*[^\W\d_].*\s[0-9]{1,3}|[0-9]{1,3}\.?)$")
_LIGATURES = str.maketrans(
    {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
)
# Some fonts leave a space after a ligature glyph: "\ufb01 ve" for "five".
_LIGATURE_GAP = re.compile(r"([\ufb00-\ufb04]) (?=[a-z])")
# Private-use code points (symbol-font bullets) and control characters.
_JUNK = re.compile(r"[\x00-\x08\x0b-\x1f\ue000-\uf8ff]")
_UNMAPPED = re.compile(r"[\x00-\x06\x0b-\x1f\ufffd]")
_CONTENTS_TITLES = {"contents", "table of contents", "content", "index"}
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


def clean_text(text: str) -> str:
    text = _LIGATURE_GAP.sub(r"\1", text).translate(_LIGATURES)
    return " ".join(_JUNK.sub(" ", text).split())


def _read_lines(pdf: Path) -> list[list[list[_Line]]]:
    """Pages, each a list of text blocks, each a list of lines."""
    pages = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            height = page.rect.height
            blocks = []
            # Unknown glyphs must remain visible as errors, not guessed CIDs.
            flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE
            for block in page.get_text("dict", flags=flags)["blocks"]:
                lines = []
                positions = []
                for line in block.get("lines", []):
                    spans = [
                        s
                        for s in line["spans"]
                        if not s["font"].lower().startswith("wingdings")
                        and s["text"].strip()
                    ]
                    if not spans:
                        continue
                    for span in spans:
                        if _UNMAPPED.search(span["text"]):
                            raise ValueError(
                                f"{pdf}: page {page.number + 1}: unmapped glyph "
                                f"in font {span['font']}: {span['text']!r}"
                            )
                    main = max(spans, key=lambda s: len(s["text"]))
                    # PDF text coordinates precede the page's display rotation.
                    rect = pymupdf.Rect(line["bbox"]) * page.rotation_matrix
                    lines.append(
                        _Line(
                            text=clean_text("".join(s["text"] for s in spans)),
                            size=round(main["size"], 1),
                            bold=all(
                                s["flags"] & 16 or "bold" in s["font"].lower()
                                for s in spans
                            ),
                            top=rect.y0 / height,
                            bottom=rect.y1 / height,
                        )
                    )
                    positions.append((rect.y0, rect.x0))
                if lines:
                    blocks.append((min(positions), lines))
            pages.append([lines for _, lines in sorted(blocks, key=lambda b: b[0])])
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

    def contents(number: int, blocks: list[list[_Line]]) -> bool:
        lines = [line.text for block in blocks for line in block]
        early = number < max(5, CONTENTS_SHARE * len(pages))
        if not early or len(lines) < 6:
            return False
        titled = any(text.lower() in _CONTENTS_TITLES for text in lines)
        listed = sum(bool(_LISTED.match(text)) for text in lines) / len(lines)
        return titled and listed >= 0.3

    return [
        []
        if contents(number, blocks)
        else [kept for block in blocks if (kept := [ln for ln in block if keep(ln)])]
        for number, blocks in enumerate(pages)
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
        return all(line.size >= body * HEADING_SCALE for line in block)

    sections = [_Section(heading="", page=1, paragraphs=[])]
    for number, blocks in enumerate(pages, start=1):
        # PDF blocks can contain both a heading and the following paragraph.
        runs = (
            list(lines)
            for block in blocks
            for _, lines in groupby(
                block, key=lambda line: line.size >= body * HEADING_SCALE
            )
        )
        for block in runs:
            block = [line for line in block if line.size >= body * LABEL_SCALE]
            if not block:
                continue
            if heading(block):
                text = " ".join(line.text for line in block)
                if text.lower() in _CONTENTS_TITLES:
                    continue  # a chapter's own "Contents" box is not a section
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
            run = ""
            for word in piece.split():
                grown = f"{run} {word}" if run else word
                if count_tokens(grown) > MAX_TOKENS:
                    if run:
                        units.append((page, run, count_tokens(run)))
                    if count_tokens(word) > MAX_TOKENS:
                        raise ValueError(
                            f"page {page}: a single word exceeds {MAX_TOKENS} tokens"
                        )
                    run = word
                else:
                    run = grown
            if run:
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
            grown = count_tokens(" ".join([*parts, units[i][1]]))
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
