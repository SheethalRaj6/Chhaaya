import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4


def test_compose_migrates_with_reserved_password_and_generates_on_host(tmp_path: Path):
    root = Path(__file__).parents[1]
    for name in (
        "Dockerfile",
        ".dockerignore",
        "compose.yaml",
        "pyproject.toml",
        "uv.lock",
        "alembic.ini",
    ):
        shutil.copy2(root / name, tmp_path / name)
    for name in ("src", "migrations"):
        shutil.copytree(root / name, tmp_path / name)

    # Synthetic credentials exercise both URL delimiters and literal percent escapes.
    password = "test@password%2F/with:colon"
    (tmp_path / ".env").write_text(
        f"POSTGRES_PASSWORD={password}\nDATABASE_URL=postgresql://unused\n"
    )
    # Check app health inside its container without reserving a host port.
    (tmp_path / "compose.test.yaml").write_text(
        "services:\n  app:\n    ports: !reset []\n"
    )
    env = {**os.environ, "POSTGRES_PASSWORD": password}
    command = [
        "docker",
        "compose",
        "--project-name",
        f"chhaaya-test-{uuid4().hex}",
        "-f",
        "compose.yaml",
        "-f",
        "compose.test.yaml",
    ]
    artifact = tmp_path / "compose.log"

    def compose(*args: str) -> str:
        with artifact.open("a") as log:
            log.write(f"$ {shlex.join([*command, *args])}\n")
            log.flush()
            result = subprocess.run(
                [*command, *args],
                cwd=tmp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )
            log.write(f"{result.stdout}\nExit status: {result.returncode}\n")
        assert result.returncode == 0, f"{result.stdout}\nArtifact: {artifact}"
        return result.stdout

    compose("version")
    try:
        compose("up", "--build", "--wait", "--wait-timeout", "120")
        health = compose(
            "exec",
            "-T",
            "app",
            "python",
            "-c",
            "from urllib.request import urlopen; "
            "print(urlopen('http://localhost:8000/health').read().decode())",
        )
        assert json.loads(health)["status"] == "ok"

        # Add a real schema change, then follow the documented generation workflow.
        models = tmp_path / "src/chhaaya/db.py"
        with models.open("a") as source:
            source.write("\nUser.review_note = mapped_column(Text, nullable=True)\n")
        compose("build", "migrate")
        compose(
            "run",
            "--rm",
            "--volume",
            "./migrations:/app/migrations",
            "migrate",
            "alembic",
            "revision",
            "--autogenerate",
            "-m",
            "add review note",
        )
        revisions = list(
            (tmp_path / "migrations/versions").glob("*_add_review_note.py")
        )
        assert len(revisions) == 1
        assert "review_note" in revisions[0].read_text()
        compose("run", "--rm", "--volume", "./migrations:/app/migrations", "migrate")
        stored_note = compose(
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            "chhaaya",
            "-d",
            "chhaaya",
            "-Atc",
            "INSERT INTO users (wa_id, role, review_note) "
            "VALUES ('synthetic-review', 'patient', 'persisted') RETURNING review_note",
        )
        assert stored_note.splitlines()[0] == "persisted"
    finally:
        compose("down", "--volumes", "--remove-orphans")
