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
import hashlib
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

# Glyph names were checked against rendered source pages. A file digest keeps
# these font-specific mappings from being applied to another PDF or edition.
GLYPH_REPAIRS = {
    "da4f59dd1aa2d31ceefb2c08b4a2013e1bc42fd229c85802761eca3f3d16a815": {
        "g308": "tt",
        "g286": "ft",
    },
    "f05769734153d3ee935060fde2ab6e92b2e39ed5f0d47e9b5c56781b81e5e51d": {
        "g332": "ft",
        "g415": "ti",
        "CambriaMath/951": "≥",
    },
}
_REPAIRED_GAP = re.compile(r"\b([A-Za-z]*(?:tt|ft|ti)) ([a-z]+)\b")
_REPAIRED_WORDS = {
    "after",
    "afterwards",
    "allotted",
    "attend",
    "attendance",
    "attendant",
    "attended",
    "attending",
    "attention",
    "attitude",
    "better",
    "bottle",
    "bottles",
    "bottom",
    "calmette",
    "committee",
    "getting",
    "ghutti",
    "little",
    "often",
    "lift",
    "time",
    "stiffness",
    "observation",
    "activities",
    "participate",
}

# Complete prose boxes checked against the posters. Algorithms, dosage tables,
# figure labels, references and contributor pages are outside these rectangles.
# Coordinates are fractions of the displayed page; digests pin each layout.
CURATED_EXCERPTS = {
    "icmr_stw_diabetes_type_2": (
        "bcad0bfca7c2ec3a10a9b695cec7b1f2f91881bad01d05d0ed3c94925ad820fb",
        [
            (1, "Symptoms", (0.18, 0.14, 0.49, 0.29)),
            (1, "Monitoring", (0.015, 0.523, 0.495, 0.605)),
            (1, "Referrals", (0.505, 0.523, 0.985, 0.605)),
        ],
    ),
    "icmr_stw_hypertension": (
        "bffc8acfa65ce92ec2d9bb914f8232b397603b8c1b0a30f1906261492a0f2ab0",
        [
            (1, "Risk factors", (0.46, 0.112, 0.57, 0.228)),
            (1, "Lifestyle modification", (0.005, 0.731, 0.25, 0.853)),
            (1, "Red flags signs", (0.765, 0.731, 0.999, 0.853)),
        ],
    ),
    "icmr_stw_fever_children": (
        "db1e43075e879b04cdc8ad471d83b73eddc9fd156578ae4a7f2240984af778ce",
        [
            (1, "Step 2: Consider hospitalization", (0.205, 0.623, 0.391, 0.829)),
            (1, "Step 3: Consider referral", (0.399, 0.623, 0.602, 0.829)),
        ],
    ),
    "icmr_stw_diarrhoea_children": (
        "22a2fcdfb18c7a5c1fce3c366090d4b34041a9ba8cd4372683e1cc7e96ca7b38",
        [
            (1, "Refer to hospital", (0.73, 0.109, 0.99, 0.271)),
            (1, "Patient education", (0.305, 0.515, 0.645, 0.582)),
            (
                1,
                "Plan A: Danger signs, return immediately",
                (0.018, 0.695, 0.29, 0.745),
            ),
        ],
    ),
    "icmr_stw_paediatric_tb": (
        "f459aff5130796ce2f3b3f952e885087e906d5e225f9a4a992c0dc197e1e08f0",
        [
            (8, "Abdominal tuberculosis: When to refer", (0.77, 0.708, 0.997, 0.88)),
            (10, "Lymph node tuberculosis: When to refer", (0.53, 0.615, 0.997, 0.728)),
            (
                11,
                "Osteoarticular tuberculosis: When to refer",
                (0.496, 0.713, 0.997, 0.798),
            ),
            (12, "Tubercular meningitis: Referral", (0.65, 0.744, 0.997, 0.873)),
        ],
    ),
}


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    publisher: str
    year: int
    url: str
    reason: str
    sha256: str = ""


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
                    sha256=row["sha256"],
                )
            )
    return sources


def download(sources: Iterable[Source], directory: Path, http: httpx2.Client) -> None:
    """Fetch each source not already in `directory`; raise naming every failure."""
    directory.mkdir(parents=True, exist_ok=True)
    failures = []
    for source in sources:
        target = directory / f"{source.id}.pdf"
        if not re.fullmatch(r"[0-9a-f]{64}", source.sha256):
            failures.append(f"{source.id}: no verified PDF checksum in sources.csv")
            continue
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != source.sha256:
                failures.append(f"{source.id}: cached PDF checksum mismatch: {target}")
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
        if hashlib.sha256(response.content).hexdigest() != source.sha256:
            failures.append(
                f"{source.id}: downloaded PDF checksum mismatch: {source.url}"
            )
            continue
        partial = target.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(target)
    if failures:
        raise DownloadError("could not download:\n  " + "\n  ".join(failures))


def chunk_source(source: Source, pdf: Path, count_tokens: TokenCounter) -> list[Chunk]:
    chunks = []
    if curated := CURATED_EXCERPTS.get(source.id):
        digest, excerpts = curated
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{source.id}: unreviewed poster edition or layout")
        sections = []
        for page, heading, bounds in excerpts:
            blocks = _read_lines(pdf, clips={page: bounds})[page - 1]
            if not blocks:
                raise ValueError(f"{source.id}: page {page}: empty excerpt {heading}")
            sections.append(
                _Section(heading, page, [(page, _join(block)) for block in blocks])
            )
    else:
        sections = _sections(_strip_furniture(_read_lines(pdf)))
    for section in sections:
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


def _read_lines(
    pdf: Path,
    *,
    clips: dict[int, tuple[float, float, float, float]] | None = None,
) -> list[list[list[_Line]]]:
    """Pages, each a list of text blocks, each a list of lines."""
    pages = []
    with pymupdf.open(pdf) as doc:
        repairs = GLYPH_REPAIRS.get(hashlib.sha256(pdf.read_bytes()).hexdigest(), {})
        _repair_fonts(doc, repairs)
        for page in doc:
            number = len(pages) + 1
            if clips is not None and number not in clips:
                pages.append([])
                continue
            height = page.rect.height
            clip = None
            if clips is not None:
                x0, y0, x1, y1 = clips[number]
                width = page.rect.width
                clip = pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
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
                        and (clip is None or clip.contains(pymupdf.Rect(s["bbox"])))
                        # The reviewed ICMR posters overlay a large diagonal
                        # Arial watermark, whose clipped letters are not prose.
                        and not (
                            clips is not None
                            and s["font"] == "Arial"
                            and s["size"] > 140
                        )
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
                    rect = pymupdf.Rect(spans[0]["bbox"])
                    for span in spans[1:]:
                        rect |= pymupdf.Rect(span["bbox"])
                    rect *= page.rotation_matrix
                    text = "".join(s["text"] for s in spans)
                    if repairs:
                        text = _REPAIRED_GAP.sub(
                            lambda m: (
                                m[1] + m[2]
                                if (m[1] + m[2]).lower() in _REPAIRED_WORDS
                                else m[0]
                            ),
                            text,
                        )
                    text = clean_text(text)
                    if not text or (clips is not None and text == "•"):
                        continue
                    lines.append(
                        _Line(
                            text=text,
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


def _repair_fonts(doc: pymupdf.Document, repairs: dict[str, str]) -> None:
    """Repair reviewed Type1 glyph mappings in memory, leaving the PDF intact."""
    if not repairs:
        return
    fonts = {
        font[0]: font[3].split("+")[-1]
        for number in range(len(doc))
        for font in doc.get_page_fonts(number)
    }
    for font, name in fonts.items():
        kind, reference = doc.xref_get_key(font, "Encoding")
        entries = []
        if kind == "xref":
            encoding = int(reference.split()[0])
            _, differences = doc.xref_get_key(encoding, "Differences")
            code = 0
            for token in re.findall(r"/[A-Za-z0-9_.]+|[0-9]+", differences):
                if token.isdecimal():
                    code = int(token)
                    continue
                if replacement := repairs.get(token[1:]):
                    value = replacement.encode("utf-16-be").hex()
                    entries.append(f"<{code:02x}> <{value}>")
                code += 1
        elif kind == "name" and reference == "/Identity-H":
            for key, replacement in repairs.items():
                if key.startswith(f"{name}/"):
                    code = int(key.split("/")[1])
                    value = replacement.encode("utf-16-be").hex()
                    entries.append(f"<{code:04x}> <{value}>")
        if not entries:
            continue
        kind, reference = doc.xref_get_key(font, "ToUnicode")
        if kind == "xref":
            cmap_xref = int(reference.split()[0])
            cmap = doc.xref_stream(cmap_xref)
        else:
            cmap_xref = doc.get_new_xref()
            doc.update_object(cmap_xref, "<<>>")
            doc.xref_set_key(font, "ToUnicode", f"{cmap_xref} 0 R")
            cmap = (
                b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
                b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) "
                b"/Supplement 0 >> def /CMapName /Chhaaya def /CMapType 2 def "
                b"1 begincodespacerange <00> <FF> endcodespacerange endcmap "
                b"CMapName currentdict /CMap defineresource pop end end"
            )
        if cmap.count(b"endcmap") != 1:
            raise ValueError(f"font {font}: malformed ToUnicode map")
        patch = (
            f"{len(entries)} beginbfchar\n" + "\n".join(entries) + "\nendbfchar\n"
        ).encode()
        doc.update_stream(cmap_xref, cmap.replace(b"endcmap", patch + b"endcmap"))


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
    partial = CHUNKS.with_suffix(".part")
    try:
        with partial.open("w", encoding="utf-8") as out:
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
        partial.replace(CHUNKS)
    finally:
        partial.unlink(missing_ok=True)
    print(f"{total} chunks from {len(sources)} sources -> {CHUNKS}", file=sys.stderr)


if __name__ == "__main__":
    main()
