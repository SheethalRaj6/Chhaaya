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

`docker compose up` starts the app, the worker and Postgres 16 with pgvector. Settings come from environment variables; `.env.example` lists them all. `GET /health` on port 8000 answers `{"status": "ok"}` once the app is up.

```bash
cp .env.example .env   # then set POSTGRES_PASSWORD
docker compose up --build
curl localhost:8000/health
```

## Database and migrations

Everything Chhaaya stores lives in one Postgres 16 database with pgvector: users (patients and ASHAs), every WhatsApp message in and out, escalation cases, medicine reminders and their doses, and the knowledge-base chunks with their 1024-dimensional embeddings. Inbound messages double as the worker's queue: each has a status the worker claims with `SELECT ... FOR UPDATE SKIP LOCKED`, and the unique WhatsApp message id means a retried webhook delivery is never stored twice. Chunks are searched by cosine similarity through an HNSW index. The tables are defined in `src/chhaaya/db.py` and built by Alembic migrations in `migrations/`; `docker compose up` applies them before the app and worker start, and the tests apply them to an empty database in a throwaway container (Docker must be running).

```bash
docker compose run --rm migrate                          # apply migrations
uv run alembic revision --autogenerate -m "add x"        # draft a migration after changing db.py
uv run pytest tests/test_migrations.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [AGENTS.md](AGENTS.md) for the rules every change and coding agent follows.

## License

AGPL-3.0. See [LICENSE](LICENSE).
