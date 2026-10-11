import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx2
import pymupdf
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from chhaaya import corpus
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


def test_early_numeric_table_survives_beside_a_contents_page(tmp_path: Path):
    rows = [f"Threshold for group {chr(65 + i)} {120 + i * 10}" for i in range(6)]
    pdf = make_pdf(
        tmp_path / "contents-and-table.pdf",
        [
            [
                ("h", "Contents"),
                *[("p", f"Chapter {chr(65 + i)} {i + 3}") for i in range(6)],
            ],
            [("h", "Blood pressure thresholds"), *[("p", row) for row in rows]],
        ],
    )

    text = " ".join(chunk.text for chunk in chunk_source(SOURCE, pdf, words))

    for row in rows:
        assert row in text
    assert "Chapter A" not in text


def test_numeric_table_cells_survive_while_margin_page_numbers_go(tmp_path: Path):
    pdf = tmp_path / "numeric-cells.pdf"
    with pymupdf.open() as doc:
        for number in range(1, 4):
            page = doc.new_page()
            page.insert_text((50, 30), HEADER, fontsize=8)
            page.insert_text((290, 820), str(number), fontsize=8)
            page.insert_text((50, 110), "Clinical thresholds", fontsize=15)
            for y, label, value in (
                (160, "Haemoglobin threshold in g/dL", "7.0"),
                (260, "Blood pressure threshold in mmHg", "180 / 120"),
            ):
                page.insert_text((50, y), label, fontsize=10)
                page.insert_text((300, y + 30), value, fontsize=5)
        doc.save(pdf)

    text = " ".join(chunk.text for chunk in chunk_source(SOURCE, pdf, words))

    assert text.count("Haemoglobin threshold in g/dL 7.0") == 3
    assert text.count("Blood pressure threshold in mmHg 180 / 120") == 3
    assert not {"1", "2", "3"} & set(text.split())


def test_subword_chunks_preserve_uneven_text_within_token_limit(tmp_path: Path):
    vocabulary = {"[UNK]": 0, "a": 1, "p": 2}
    vocabulary.update(
        {f"##{letter}": i + 3 for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
    )
    tokenizer = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    text = " ".join(["pneumonoultramicroscopic"] * 60 + ["a"] * 250)
    pdf = make_pdf(tmp_path / "uneven-tokens.pdf", [[("h", "Vocabulary"), ("p", text)]])

    chunks = [
        chunk
        for chunk in chunk_source(SOURCE, pdf, count)
        if chunk.section == "Vocabulary"
    ]

    assert " ".join(chunk.text for chunk in chunks) == text
    assert len(chunks) >= 4
    assert all(chunk.tokens == count(chunk.text) <= 500 for chunk in chunks)


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("caption", [False, True])
def test_clinical_tables_keep_merged_conditions_and_column_labels(
    tmp_path, rotation, caption
):
    pdf = tmp_path / "clinical-table.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 110), "Clinical thresholds", fontsize=15)
        if caption:
            page.draw_rect(pymupdf.Rect(50, 130, 550, 150))
            page.insert_text(
                (60, 145),
                "Interpret the thresholds with the group and severity.",
                fontsize=10,
            )
        for x in (50, 250, 400, 550):
            page.draw_line((x, 150), (x, 300))
        for y in (150, 180, 260, 300):
            page.draw_line((50, y), (550, y))
        page.draw_line((250, 220), (550, 220))
        for x, y, text in (
            (60, 170, "Group"),
            (260, 170, "Severity"),
            (410, 170, "Threshold"),
            (60, 200, "Children"),
            (260, 200, "Moderate"),
            (410, 200, "7-9.9"),
            (260, 240, "Severe"),
            (410, 240, "<7"),
            (60, 280, "Adults"),
            (260, 280, "Severe"),
            (410, 280, "<8"),
        ):
            page.insert_text((x, y), text, fontsize=10)
        doc.save(pdf)
    if rotation:
        rotated = tmp_path / "sideways-table.pdf"
        with pymupdf.open(pdf) as source, pymupdf.open() as doc:
            page = doc.new_page(width=842, height=595)
            page.show_pdf_page(page.rect, source, 0, rotate=rotation)
            doc.save(rotated)
        pdf = rotated

    chunks = chunk_source(SOURCE, pdf, words)
    text = " ".join(c.text for c in chunks)
    assert "Group: Children; Severity: Moderate; Threshold: 7-9.9" in text
    assert "Group: Children; Severity: Severe; Threshold: <7" in text
    assert "Group: Adults; Severity: Severe; Threshold: <8" in text
    assert {c.section for c in chunks} == {"Clinical thresholds"}


def test_rotated_pages_keep_visits_in_reading_order(tmp_path: Path):
    original = tmp_path / "upright.pdf"
    with pymupdf.open() as doc:
        for number in range(1, 4):
            page = doc.new_page()
            page.insert_text((50, 30), HEADER, fontsize=8)
            page.insert_text((290, 820), str(number), fontsize=8)
            page.insert_text((50, 140), "Home visits", fontsize=15)
            for y, text in (
                (180, "First visit: registration."),
                (230, "Second visit: follow up."),
                (280, "Third visit: review."),
            ):
                page.insert_text((50, y), text, fontsize=10)
        doc.save(original)
    rotated = tmp_path / "rotated.pdf"
    with pymupdf.open(original) as source, pymupdf.open() as doc:
        for number, original_page in enumerate(source):
            page = doc.new_page(
                width=original_page.rect.height, height=original_page.rect.width
            )
            page.show_pdf_page(page.rect, source, number, rotate=90)
            page.set_rotation(90)
        doc.save(rotated)

    chunks = chunk_source(SOURCE, rotated, words)
    text = " ".join(chunk.text for chunk in chunks)

    assert (
        text.count(
            "First visit: registration. Second visit: follow up. Third visit: review."
        )
        == 3
    )
    assert {chunk.section for chunk in chunks} == {"Home visits"}
    assert "Skills that Save Lives" not in text


def test_long_table_cells_repeat_the_condition_on_every_continuation(tmp_path):
    pdf = tmp_path / "long-table.pdf"
    advice = " ".join(sentences("feeding", 30))
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 110), "Feeding advice", fontsize=15)
        for x in (50, 200, 550):
            page.draw_line((x, 150), (x, 750))
        for y in (150, 180, 750):
            page.draw_line((50, y), (550, y))
        page.insert_text((60, 170), "Group", fontsize=10)
        page.insert_text((210, 170), "Advice", fontsize=10)
        page.insert_text((60, 200), "Children", fontsize=10)
        assert (
            page.insert_textbox(pymupdf.Rect(210, 190, 540, 740), advice, fontsize=8)
            >= 0
        )
        doc.save(pdf)
    chunks = chunk_source(SOURCE, pdf, words)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "Group: Children" in chunk.text
        assert "Advice:" in chunk.text
        assert chunk.tokens <= MAX_TOKENS
    text = " ".join(c.text for c in chunks)
    assert "Feeding note aa tells" in text
    assert "Feeding note bd tells" in text


def test_symbol_font_bullets_do_not_become_letters_in_advice(tmp_path: Path):
    pdf = tmp_path / "symbol-bullet.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 110), "Care advice", fontsize=15)
        page.insert_text((50, 150), "z", fontname="cour", fontsize=10)
        page.insert_text((60, 150), "Keep breastfeeding.", fontsize=10)
        font = next(f[0] for f in doc.get_page_fonts(0) if f[4] == "cour")
        doc.xref_set_key(font, "BaseFont", "/Wingdings-Regular")
        doc.save(pdf)

    chunks = chunk_source(SOURCE, pdf, words)

    assert [chunk.text for chunk in chunks] == ["Keep breastfeeding."]


def test_unmapped_glyphs_fail_instead_of_corrupting_clinical_text(tmp_path: Path):
    pdf = tmp_path / "unmapped-glyph.pdf"
    font = pymupdf.Font("helv")
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_font(fontname="clinical", fontbuffer=font.buffer)
        page.insert_text(
            (50, 140), "After care matters.", fontname="clinical", fontsize=10
        )
        font_xref = doc.get_page_fonts(0)[0][0]
        cmap_xref = int(doc.xref_get_key(font_xref, "ToUnicode")[1].split()[0])
        cmap = doc.xref_stream(cmap_xref)
        # The drawn glyph remains 'f', but the broken Unicode map loses it.
        override = (
            f"1 beginbfchar\n<{font.has_glyph(ord('f')):04x}> <0004>\n"
            "endbfchar\nendcmap"
        ).encode()
        doc.update_stream(cmap_xref, cmap.replace(b"endcmap", override))
        doc.save(pdf)

    with pytest.raises(ValueError, match="unmapped-glyph.pdf.*page 1"):
        chunk_source(SOURCE, pdf, words)


def test_verified_font_repairs_apply_only_to_the_reviewed_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "reviewed.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 140), "Ofen review matters.", fontsize=10)
        font = doc.get_page_fonts(0)[0][0]
        encoding = doc.get_new_xref()
        doc.update_object(
            encoding,
            "<< /Type /Encoding /BaseEncoding /WinAnsiEncoding "
            "/Differences [102 /gMissing] >>",
        )
        doc.xref_set_key(font, "Encoding", f"{encoding} 0 R")
        cmap = doc.get_new_xref()
        doc.update_object(cmap, "<<>>")
        doc.update_stream(
            cmap,
            b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) "
            b"/Supplement 0 >> def /CMapName /test def /CMapType 2 def "
            b"1 begincodespacerange <00> <FF> endcodespacerange "
            b"1 beginbfchar <66> <FFFD> endbfchar endcmap "
            b"CMapName currentdict /CMap defineresource pop end end",
        )
        doc.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        doc.save(pdf)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(
        corpus, "GLYPH_REPAIRS", {digest: {"gMissing": "ft"}}, raising=False
    )

    chunks = chunk_source(SOURCE, pdf, words)
    assert [chunk.text for chunk in chunks] == ["Often review matters."]

    changed = tmp_path / "different.pdf"
    with pymupdf.open(pdf) as doc:
        doc.set_metadata({"title": "Different edition"})
        doc.save(changed)
    with pytest.raises(ValueError, match="different.pdf.*page 1"):
        chunk_source(SOURCE, changed, words)


def test_reviewed_spacing_repairs_preserve_real_word_boundaries(tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "spaced-ligatures.pdf", [[
        ("h", "Antenatal care"),
        ("p", "Th is is the fi rst visit aft er registration. Check soft tissue with the mother."),
    ]])
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(corpus, "SPACING_REPAIRS", {digest}, raising=False)
    text = " ".join(c.text for c in chunk_source(SOURCE, pdf, words))
    assert "This is the first visit after registration." in text
    assert "Check soft tissue with the mother." in text


def test_reviewed_flowchart_region_does_not_linearize_branches(tmp_path, monkeypatch):
    pdf = tmp_path / "prose-and-flowchart.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 110), "Referral guidance", fontsize=15)
        page.insert_textbox(pymupdf.Rect(50, 150, 300, 280),
                            "If the child cannot drink, seek urgent care.", fontsize=10)
        for box, text in [((320, 150, 550, 180), "Evaluate child"),
                          ((320, 230, 420, 270), "YES: Admit"),
                          ((450, 230, 550, 270), "NO: Discharge")]:
            rect = pymupdf.Rect(box)
            page.draw_rect(rect)
            page.insert_text((rect.x0 + 5, rect.y0 + 15), text, fontsize=8)
        page.draw_line((375, 180), (375, 230))
        page.draw_line((375, 200), (500, 200))
        page.draw_line((500, 200), (500, 230))
        doc.save(pdf)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(corpus, "EXCLUDED_REGIONS", {digest: {1: (0.5, 0.15, 1.0, 0.52)}}, raising=False)
    text = " ".join(c.text for c in chunk_source(SOURCE, pdf, words))
    assert "If the child cannot drink, seek urgent care." in text
    assert "Evaluate child" not in text
    assert "Admit" not in text
    assert "Discharge" not in text


def test_repeated_clinical_table_rows_in_page_margins_are_not_furniture(tmp_path):
    pdf = tmp_path / "margin-table.pdf"
    with pymupdf.open() as doc:
        for _ in range(3):
            page = doc.new_page()
            page.insert_text((50, 150), "Clinical thresholds", fontsize=15)
            for x in (50, 250, 550):
                page.draw_line((x, 720), (x, 810))
            for y in (720, 750, 810):
                page.draw_line((50, y), (550, y))
            for x, y, text in [(60, 740, "Group"), (260, 740, "Threshold"),
                               (60, 780, "Children"), (260, 780, "<7")]:
                page.insert_text((x, y), text, fontsize=10)
        doc.save(pdf)
    text = " ".join(c.text for c in chunk_source(SOURCE, pdf, words))
    assert text.count("Group: Children; Threshold: <7") == 3


def test_failed_build_preserves_previous_complete_corpus(tmp_path: Path, monkeypatch):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    make_pdf(
        pdfs / "asha_module_x.pdf",
        [[("h", "Clinical care"), *paragraphs("care", 8)]],
    )
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(pdfs / "scanned.pdf")
    scanned = Source("scanned", "Scanned", "MoHFW", 2020, SOURCE.url, "test")
    scanned = replace(
        scanned, sha256=hashlib.sha256((pdfs / "scanned.pdf").read_bytes()).hexdigest()
    )
    source = replace(
        SOURCE,
        sha256=hashlib.sha256((pdfs / "asha_module_x.pdf").read_bytes()).hexdigest(),
    )
    output = tmp_path / "chunks.jsonl"
    previous = '{"id":"previous:1","text":"Previously verified corpus"}\n'
    output.write_text(previous)
    monkeypatch.setattr(corpus, "PDFS", pdfs)
    monkeypatch.setattr(corpus, "CHUNKS", output)
    monkeypatch.setattr(corpus, "read_sources", lambda: [source, scanned])
    monkeypatch.setattr(corpus, "_bge_m3_counter", lambda: words)

    with pytest.raises(SystemExit, match="scanned: no text extracted"):
        corpus.main()

    assert output.read_text() == previous
    assert not output.with_suffix(".part").exists()

    monkeypatch.setattr(corpus, "read_sources", lambda: [source])
    corpus.main()
    built = [json.loads(line) for line in output.read_text().splitlines()]
    assert {chunk["source_id"] for chunk in built} == {SOURCE.id}
    assert "Care note aa" in " ".join(chunk["text"] for chunk in built)


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
    good = replace(good, sha256=hashlib.sha256(b"%PDF-1.7 fake").hexdigest())
    moved = replace(moved, sha256=good.sha256)
    kept = replace(kept, sha256=hashlib.sha256(b"%PDF-1.7 already here").hexdigest())

    with pytest.raises(DownloadError, match="moved"):
        download([good, moved, kept], tmp_path, http)

    assert (tmp_path / "good.pdf").read_bytes() == b"%PDF-1.7 fake"
    assert not (tmp_path / "moved.pdf").exists()
    assert "https://example.org/kept.pdf" not in requested


def test_download_rejects_a_wrong_cached_edition_and_wrong_remote_bytes(tmp_path):
    verified = b"%PDF-1.7 reviewed edition"
    source = replace(SOURCE, sha256=hashlib.sha256(verified).hexdigest())
    cached = tmp_path / f"{source.id}.pdf"
    cached.write_bytes(b"%PDF-1.7 unrelated book")
    http = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, content=b"%PDF-1.7 wrong edition")
        )
    )

    with pytest.raises(DownloadError, match="checksum"):
        download([source], tmp_path, http)
    assert cached.read_bytes() == b"%PDF-1.7 unrelated book"
    cached.unlink()
    with pytest.raises(DownloadError, match="checksum"):
        download([source], tmp_path, http)
    assert not cached.exists()

    http = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, content=verified)
        )
    )
    download([source], tmp_path, http)
    assert cached.read_bytes() == verified


def test_curated_poster_excerpts_keep_context_and_exclude_flowcharts(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "poster.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 120), "When to refer", fontsize=15)
        page.insert_text((50, 150), "If the child cannot drink, refer to hospital.")
        page.insert_text((550, 170), "Figure title", fontsize=11)
        page.insert_text((50, 450), "YES: Give first line treatment.")
        page.insert_text((350, 550), "NO: Further investigation.")
        page.draw_line((50, 470), (350, 530))
        doc.save(pdf)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    monkeypatch.setattr(
        corpus,
        "CURATED_EXCERPTS",
        {SOURCE.id: (digest, [(1, "When to refer", (0.05, 0.12, 0.95, 0.22))])},
        raising=False,
    )

    chunks = chunk_source(SOURCE, pdf, words)
    assert [(c.page, c.section, c.text) for c in chunks] == [
        (
            1,
            "When to refer",
            "When to refer If the child cannot drink, refer to hospital.",
        )
    ]

    changed = tmp_path / "changed-poster.pdf"
    with pymupdf.open(pdf) as doc:
        doc.set_metadata({"title": "New layout"})
        doc.save(changed)
    with pytest.raises(ValueError, match="unreviewed poster"):
        chunk_source(SOURCE, changed, words)


def test_every_source_is_listed_with_what_a_citation_needs():
    with SOURCES.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == ["id", "title", "publisher", "year", "url", "reason", "sha256"]

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
