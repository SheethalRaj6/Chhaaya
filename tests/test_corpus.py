import csv
from pathlib import Path

import httpx2
import pymupdf
import pytest

from chhaaya.corpus import (
    MAX_TOKENS,
    MIN_TOKENS,
    SOURCES,
    DownloadError,
    Source,
    chunk_source,
    clean_text,
    download,
    read_sources,
)

SOURCE = Source(
    id="asha_module_x",
    title="ASHA Module X",
    publisher="NHM, MoHFW",
    year=2011,
    url="https://example.org/asha-module-x.pdf",
    reason="test",
)
HEADER = "ASHA Module X | Skills that Save Lives"


def words(text: str) -> int:
    """Stands in for the BGE-M3 tokenizer, which needs a download."""
    return len(text.split())


def sentences(topic: str, count: int) -> list[str]:
    return [
        f"{topic.capitalize()} note {chr(97 + i // 26)}{chr(97 + i % 26)} tells the "
        f"family what to watch for and when to visit the health centre about {topic}."
        for i in range(count)
    ]


def make_pdf(path: Path, pages: list[list[tuple[str, str]]]) -> Path:
    """Each page is a list of (kind, text).

    kind is "h" for a heading, "b" for a bold line in body size, "l" for a
    tiny figure label, or "p" for a paragraph.
    """
    doc = pymupdf.open()
    for number, blocks in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 30), HEADER, fontsize=8)
        page.insert_text((290, 820), str(number), fontsize=8)
        y = 60.0
        for kind, text in blocks:
            if kind == "h":
                page.insert_text((50, y + 14), text, fontsize=15, fontname="hebo")
                y += 30
            elif kind == "b":
                page.insert_text((50, y + 10), text, fontsize=10, fontname="hebo")
                y += 20
            elif kind == "l":
                page.insert_text((300, y + 5), text, fontsize=5)
                y += 12
            else:
                rect = pymupdf.Rect(50, y, 545, 790)
                left = page.insert_textbox(rect, text, fontsize=10, fontname="helv")
                assert left >= 0, "test paragraph does not fit on the page"
                y = 790 - left + 8
    doc.save(path)
    return path


def paragraphs(topic: str, count: int, per_paragraph: int = 6) -> list[tuple[str, str]]:
    lines = sentences(topic, count)
    return [
        ("p", " ".join(lines[i : i + per_paragraph]))
        for i in range(0, count, per_paragraph)
    ]


@pytest.fixture
def module(tmp_path: Path) -> Path:
    contents = [
        ("h", "Contents"),
        ("p", "1. Fever ........................ 2"),
        ("p", "2. Diarrhoea ..................... 4"),
    ]
    fever = paragraphs("fever", 66)
    diarrhoea = paragraphs("diarrhoea", 30)
    return make_pdf(
        tmp_path / "module.pdf",
        [
            contents,
            [("h", "1. Fever"), *fever[:6]],
            fever[6:],
            [("h", "2. Diarrhoea"), *diarrhoea],
            [("h", "Danger signs"), *paragraphs("danger", 4)],
        ],
    )


def test_running_headers_page_numbers_and_contents_are_removed(module: Path):
    text = "\n".join(c.text for c in chunk_source(SOURCE, module, words))

    assert "Skills that Save Lives" not in text
    assert "...." not in text
    for page_number in ("1", "2", "3", "4", "5"):
        assert page_number not in text.split()
    assert "Fever note aa tells the family" in text
    assert "Danger note ad tells the family" in text


def test_chunks_never_cross_a_section_and_stay_within_size(module: Path):
    chunks = chunk_source(SOURCE, module, words)

    topics = [
        {t for t in ("fever", "diarrhoea", "danger") if t in c.text} for c in chunks
    ]
    assert all(len(t) == 1 for t in topics)
    sections = {c.section for c in chunks}
    assert {"1. Fever", "2. Diarrhoea", "Danger signs"} <= sections

    for chunk in chunks:
        assert chunk.tokens == words(chunk.text) <= MAX_TOKENS
    # Fever is long enough to need several chunks, and all of them are full size.
    fever = [c for c in chunks if c.section == "1. Fever"]
    assert len(fever) > 1
    assert all(c.tokens >= MIN_TOKENS for c in fever)


def test_headings_sharing_a_block_with_body_text_still_separate_topics(tmp_path: Path):
    pdf = tmp_path / "mixed-blocks.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        for topic, y in (("Fever", 110), ("Diarrhoea", 370)):
            page.insert_text((50, y), topic, fontsize=15)
            page.insert_textbox(
                pymupdf.Rect(50, y + 3, 545, y + 220),
                " ".join(sentences(topic.lower(), 8)),
                fontsize=10,
            )
        doc.save(pdf)

    chunks = chunk_source(SOURCE, pdf, words)

    assert {chunk.section for chunk in chunks} == {"Fever", "Diarrhoea"}
    for chunk in chunks:
        assert chunk.section.lower() in chunk.text.lower()
        other = "diarrhoea" if chunk.section == "Fever" else "fever"
        assert other not in chunk.text.lower()


def test_every_chunk_keeps_its_source_and_the_page_it_starts_on(module: Path):
    chunks = chunk_source(SOURCE, module, words)

    for chunk in chunks:
        assert (chunk.title, chunk.publisher, chunk.year, chunk.url) == (
            "ASHA Module X",
            "NHM, MoHFW",
            2011,
            "https://example.org/asha-module-x.pdf",
        )
    first = {c.section: c.page for c in reversed(chunks)}
    assert first["1. Fever"] == 2
    assert first["2. Diarrhoea"] == 4
    assert first["Danger signs"] == 5
    # The fever section runs onto page 3, and a later chunk starts there.
    assert {c.page for c in chunks if c.section == "1. Fever"} == {2, 3}


def test_download_fetches_missing_pdfs_and_refuses_anything_else(tmp_path: Path):
    requested: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requested.append(str(request.url))
        if request.url.path.endswith("good.pdf"):
            return httpx2.Response(200, content=b"%PDF-1.7 fake")
        return httpx2.Response(200, content=b"<html>Page moved</html>")

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    good = Source("good", "Good", "MoHFW", 2020, "https://example.org/good.pdf", "-")
    moved = Source(
        "moved", "Moved", "MoHFW", 2020, "https://example.org/moved.pdf", "-"
    )
    kept = Source("kept", "Kept", "MoHFW", 2020, "https://example.org/kept.pdf", "-")
    (tmp_path / "kept.pdf").write_bytes(b"%PDF-1.7 already here")

    with pytest.raises(DownloadError, match="moved"):
        download([good, moved, kept], tmp_path, http)

    assert (tmp_path / "good.pdf").read_bytes() == b"%PDF-1.7 fake"
    assert not (tmp_path / "moved.pdf").exists()
    assert "https://example.org/kept.pdf" not in requested


def test_every_source_is_listed_with_what_a_citation_needs():
    with SOURCES.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["id", "title", "publisher", "year", "url", "reason"]

    sources = read_sources(SOURCES)
    assert len({s.id for s in sources}) == len(sources)
    for source in sources:
        assert source.title and source.publisher and source.reason
        assert 1990 <= source.year <= 2030
        assert source.url.startswith("https://")


def test_a_row_with_an_unquoted_comma_is_rejected_not_truncated(tmp_path: Path):
    path = tmp_path / "sources.csv"
    path.write_text(
        "id,title,publisher,year,url,reason\n"
        "sba,SBA,MoHFW,2010,https://example.org/ANMs,LHVs,SNs.pdf,antenatal care\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        read_sources(path)


def test_bold_subheadings_stay_in_their_section_and_figure_labels_go(tmp_path: Path):
    pdf = make_pdf(
        tmp_path / "boxes.pdf",
        [
            [
                ("h", "6. Management of Diarrhoea"),
                *paragraphs("diarrhoea", 12),
                ("b", "Remember:"),
                ("p", "Give ORS after every loose stool and keep breastfeeding."),
                ("l", "FIRST VISIT"),
                ("h", "Contents"),
                ("b", "Tips to help the mother:"),
                *paragraphs("feeding", 6),
            ],
            # Two more pages so the running header repeats and is removed.
            paragraphs("feeding", 12),
            paragraphs("feeding", 12),
        ],
    )

    chunks = chunk_source(SOURCE, pdf, words)

    assert {c.section for c in chunks} == {"6. Management of Diarrhoea"}
    text = " ".join(c.text for c in chunks)
    assert "Remember: Give ORS after every loose stool" in text
    assert "Tips to help the mother:" in text
    assert "FIRST VISIT" not in text


def test_split_ligatures_and_symbol_font_bullets_are_cleaned():
    assert (
        clean_text("\uf0b7 the \ufb01 rst speci\ufb01 c visit")
        == "the first specific visit"
    )
    assert clean_text("o\ufb00 er e\ufb03 cient") == "offer efficient"
