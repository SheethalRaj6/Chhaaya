# Chhaaya

Chhaaya is a WhatsApp health assistant for rural patients in India. You send it a voice note, a message or a photo, in Hindi, English or Hinglish. It answers from Indian public-health guidelines, explains your lab report in plain words, reminds you to take your medicines, and passes anything urgent or uncertain to your ASHA worker. It does not diagnose.

The project is at an early stage: the service skeleton runs, the features do not exist yet. [docs/design.md](docs/design.md) is the specification, and the open issues are the work plan.

## How it works

A message arrives through the WhatsApp Cloud API and is queued in Postgres. A worker then:

1. turns voice into text with Sarvam's speech-to-text;
2. checks the message against a fixed list of danger signs;
3. routes it with a small intent classifier we fine-tune from MuRIL.

Questions are answered by retrieving passages from MoHFW, ICMR and NHM documents and having an LLM answer only from those passages. Any answer that cites nothing it was given is thrown away. Lab reports are read with OCR and compared against the ranges printed on them. Prescriptions are confirmed by the ASHA before the patient hears anything about them. Replies go back as text and as a voice note.

## Running it locally

`docker compose up` starts the app, the worker and Postgres 16 with pgvector. Settings come from environment variables; `.env.example` lists them all. The service refuses to start with a blank `WHATSAPP_*` value; without a Meta app, any non-blank placeholders work locally, but never deploy with guessable ones. `GET /health` on port 8000 answers `{"status": "ok"}` once the app is up.

```bash
cp .env.example .env   # then set POSTGRES_PASSWORD and the WHATSAPP_* values
docker compose up --build
curl localhost:8000/health
```

## Database and migrations

Everything Chhaaya stores lives in one Postgres 16 database with pgvector: users (patients and ASHAs), every WhatsApp message in and out, escalation cases, medicine reminders and their doses, and the knowledge-base chunks with their 1024-dimensional embeddings. Inbound messages double as the worker's queue: each has a status the worker claims with `SELECT ... FOR UPDATE SKIP LOCKED`, and the unique WhatsApp message id means a retried webhook delivery is never stored twice. Chunks are searched by cosine similarity through an HNSW index. The tables are defined in `src/chhaaya/db.py` and built by Alembic migrations in `migrations/`; `docker compose up` applies them before the app and worker start, and the tests apply them to an empty database in a throwaway container (Docker must be running). Compose passes the password separately through `PGPASSWORD`, preserving characters that have special meanings in URLs. After changing a model, rebuild the migration image and generate the revision inside Compose, where Postgres is reachable; the bind mount saves the new migration in your checkout. Review the generated file before applying it.

```bash
docker compose build migrate
docker compose run --rm --volume ./migrations:/app/migrations migrate alembic revision --autogenerate -m "add x"
docker compose run --rm --volume ./migrations:/app/migrations migrate  # apply migrations
uv run pytest tests/test_migrations.py tests/test_compose.py
```

## Danger-sign detection

Every message is screened for danger signs by rules rather than a model, so anyone can read why a message was flagged. `data/danger_signs.yaml` lists each sign (for example bleeding in pregnancy, convulsions, a baby not feeding, difficulty breathing, unconsciousness) with the page of the MCP card, IMNCI or ASHA module it comes from, its phrasings in Hindi, English and Hinglish, and the urgent reply (go to the nearest health centre or call 108) in all three. `chhaaya.danger_signs.detect(text)` folds case, Unicode forms and common Hinglish spellings ("nhi", "dudh", "rha"), then looks for each phrasing in order with up to six other words between its words, so "pet mein bahut tez dard" still matches "pet tez dard". A negated mention such as "no bleeding" or "pregnancy mein khoon nahi aaya" does not fire, but because a missed sign is far worse than a false alarm, negation only suppresses a match in those narrow patterns: "pregnancy mein bleeding nahi ruk rahi" (the bleeding won't stop) fires. Wiring it into the worker comes with the worker itself (issue #10).

```bash
uv run python -c 'from chhaaya.danger_signs import detect; print(detect("bachcha doodh nahi pee raha"))'
uv run pytest tests/test_danger_signs.py
```

## Knowledge corpus

The Q&A agent answers only from a small set of Indian public-health documents, so they are chosen by hand. `data/sources.csv` lists each one (NHM ASHA modules and handbook, MoHFW guidelines on diarrhoea, immunisation, pregnancy, NCDs and TB, and ICMR Standard Treatment Workflows) with its title, publisher, year, the URL it was fetched from, why it is in, and the SHA-256 of the reviewed PDF. `chhaaya.corpus` downloads the PDFs to `data/pdfs/` and writes `data/chunks.jsonl`; neither is committed. A PDF is accepted, whether downloaded or already in `data/pdfs/`, only if its checksum matches the CSV, so the per-PDF fixes below never touch another edition. It reads each page with PyMuPDF, turns sideways pages upright, and drops running headers and footers (margin lines repeated on many pages), page numbers, contents pages, symbol-font bullets and small figure labels; a small line that contains a number stays, because it can be a dose or a value. Ligatures are re-joined ("ﬁ ve" becomes "five"), and in three PDFs whose fonts map glyphs wrongly, reviewed glyph and word-spacing repairs apply; an unmapped glyph stops the build instead of reaching a chunk. Tables become one line per row, "Table row: caption; header: value; header: value", so each value keeps its condition and column label; a row too long for one chunk is split on its largest cell and repeats every other cell. Reviewed pages are left out by page: flowcharts that only make sense as drawings, a quiz with wrong answers, blank planning and reporting forms, and one infant-positioning table the parser cannot split. From the ICMR posters only reviewed prose boxes (symptoms, risk factors, lifestyle advice, monitoring, danger signs and when to refer) are kept, not their algorithms or dose tables. A new section starts at each heading set in a font larger than the body text; bold body-size lines such as "Remember:" stay inside their section. Each section is cut into chunks of 300 to 500 tokens, counted with the BGE-M3 tokenizer the embeddings will use, at paragraph or sentence boundaries; a chunk never crosses a section, so a short section stays one smaller chunk. Every chunk keeps the document's title, publisher, year and URL, its section heading, and the PDF page it starts on. `data/chunks.jsonl` is replaced only once every source has been chunked, so a failed build leaves the previous one in place. The first run downloads the tokenizer from Hugging Face. Embedding and loading the chunks is issue #5.

```bash
uv run python -m chhaaya.corpus          # download what's missing, then rebuild data/chunks.jsonl
uv run pytest tests/test_corpus.py
```

## WhatsApp webhook

Messages reach Chhaaya through the WhatsApp Cloud API. `POST /webhook` checks Meta's `X-Hub-Signature-256` signature against the app secret and rejects anything unsigned or forged. It stores each text, voice note and photo as a pending message keyed by its WhatsApp message id, so a retried delivery is ignored, and answers `200` straight away; the worker does the rest. `GET /webhook` answers Meta's verification challenge. `src/chhaaya/whatsapp.py` is the one client for sending text and audio, uploading media and downloading what patients send.

To connect a test number:

1. At developers.facebook.com create a Business app and add the WhatsApp product. API Setup gives you a test number: copy its phone number id and a temporary access token (valid 24 hours).
2. Under App settings, Basic, copy the app secret.
3. In API Setup add up to five recipient phones and confirm each with the code WhatsApp sends. The test number can only message these.
4. Put the four values in `.env`: `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_APP_SECRET`, and a `WHATSAPP_VERIFY_TOKEN` of your choosing.
5. In Cloudflare Zero Trust create a tunnel, add a public hostname that points to `http://app:8000`, and put its token in `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.
6. In the Meta app, WhatsApp, Configuration, set the callback URL to `https://<your-hostname>/webhook` and the verify token to your `WHATSAPP_VERIFY_TOKEN`, then subscribe to the `messages` field.

```bash
docker compose --profile tunnel up --build
```

Send a text, a voice note and a photo from a recipient phone; each should appear once in the `messages` table.

```bash
docker compose exec db psql -U chhaaya -c "SELECT wa_message_id, kind, status FROM messages"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [AGENTS.md](AGENTS.md) for the rules every change and coding agent follows.

## License

AGPL-3.0. See [LICENSE](LICENSE).
