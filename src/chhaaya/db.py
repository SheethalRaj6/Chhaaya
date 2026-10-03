"""Database tables. The Alembic migrations in `migrations/` must build exactly these."""

from datetime import date, datetime, time

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Text,
    Time,
    UniqueConstraint,
    func,
    make_url,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSIONS = 1024  # BGE-M3


def sqlalchemy_url(database_url: str) -> URL:
    """Name the psycopg 3 driver in a plain postgresql:// DATABASE_URL."""
    return make_url(database_url).set(drivername="postgresql+psycopg")


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        int: BigInteger,
        str: Text,
    }


class User(Base):
    """A patient or an ASHA, identified by their WhatsApp id (phone number)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    wa_id: Mapped[str] = mapped_column(unique=True)
    role: Mapped[str]
    # Hindi is ("hi", "Deva"), Hinglish ("hi", "Latn") and English ("en", "Latn").
    language: Mapped[str | None]
    script: Mapped[str | None]
    asha_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('patient', 'asha')", name="role"),
        CheckConstraint("language IN ('hi', 'en')", name="language"),
        CheckConstraint("script IN ('Deva', 'Latn')", name="script"),
        CheckConstraint(
            "role = 'patient' OR asha_id IS NULL", name="only_patients_have_an_asha"
        ),
    )


class Message(Base):
    """Every WhatsApp message in or out. Inbound messages are the worker's queue."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    wa_message_id: Mapped[str] = mapped_column(unique=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    direction: Mapped[str]
    kind: Mapped[str]
    body: Mapped[str | None]
    media_id: Mapped[str | None]
    # Inbound only: pending -> processing (claimed with FOR UPDATE SKIP LOCKED)
    # -> done or failed.
    status: Mapped[str | None]
    claimed_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("direction IN ('inbound', 'outbound')", name="direction"),
        CheckConstraint("kind IN ('text', 'audio', 'image', 'template')", name="kind"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')", name="status"
        ),
        CheckConstraint(
            "(direction = 'inbound') = (status IS NOT NULL)",
            name="only_inbound_has_status",
        ),
        Index("ix_messages_pending", "id", postgresql_where="status = 'pending'"),
    )


class Case(Base):
    """An escalation to the patient's ASHA. The ASHA replies quoting its number."""

    __tablename__ = "cases"

    number: Mapped[int] = mapped_column(Identity(), primary_key=True)
    cause: Mapped[str]
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    asha_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(server_default="open")
    asha_reply: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    answered_at: Mapped[datetime | None]

    __table_args__ = (
        CheckConstraint(
            "cause IN ('danger_sign', 'weak_evidence', 'citation_failed', "
            "'critical_lab_value', 'prescription_confirmation', 'missed_doses')",
            name="cause",
        ),
        CheckConstraint("status IN ('open', 'answered')", name="status"),
        CheckConstraint(
            "(status = 'answered') = "
            "(asha_reply IS NOT NULL AND answered_at IS NOT NULL)",
            name="answered_has_reply",
        ),
    )


class Reminder(Base):
    """A medicine taken at fixed IST times each day, from starts_on to ends_on."""

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    medicine: Mapped[str]
    times: Mapped[list[time]] = mapped_column(ARRAY(Time))
    starts_on: Mapped[date] = mapped_column(Date)
    ends_on: Mapped[date | None] = mapped_column(Date)
    # NULL once the course has ended; the worker polls this for due reminders.
    next_due_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint("cardinality(times) > 0", name="has_times"),
        CheckConstraint("ends_on >= starts_on", name="ends_after_start"),
        Index(
            "ix_reminders_next_due_at",
            "next_due_at",
            postgresql_where="next_due_at IS NOT NULL",
        ),
    )


class Dose(Base):
    """One reminder sent at one due time, and whether the patient took it."""

    __tablename__ = "doses"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    reminder_id: Mapped[int] = mapped_column(
        ForeignKey("reminders.id", ondelete="CASCADE")
    )
    due_at: Mapped[datetime]
    status: Mapped[str] = mapped_column(server_default="pending")
    taken_at: Mapped[datetime | None]

    __table_args__ = (
        UniqueConstraint("reminder_id", "due_at"),
        CheckConstraint("status IN ('pending', 'taken', 'missed')", name="status"),
        CheckConstraint(
            "(status = 'taken') = (taken_at IS NOT NULL)", name="taken_has_time"
        ),
    )


class KbChunk(Base):
    """A retrievable passage: a guideline page, or an ASHA's earlier answer."""

    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    text: Mapped[str]
    title: Mapped[str]
    publisher: Mapped[str | None]
    year: Mapped[int | None] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str | None]
    source_type: Mapped[str]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('guideline', 'health_worker_answer')", name="source_type"
        ),
        CheckConstraint(
            "source_type = 'health_worker_answer' OR (publisher IS NOT NULL "
            "AND year IS NOT NULL AND page IS NOT NULL AND url IS NOT NULL)",
            name="guideline_is_cited",
        ),
        Index(
            "ix_kb_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
