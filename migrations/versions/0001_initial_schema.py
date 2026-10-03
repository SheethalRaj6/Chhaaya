"""initial schema

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("wa_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("script", sa.Text(), nullable=True),
        sa.Column("asha_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("language IN ('hi', 'en')", name=op.f("ck_users_language")),
        sa.CheckConstraint(
            "role = 'patient' OR asha_id IS NULL",
            name=op.f("ck_users_only_patients_have_an_asha"),
        ),
        sa.CheckConstraint("role IN ('patient', 'asha')", name=op.f("ck_users_role")),
        sa.CheckConstraint("script IN ('Deva', 'Latn')", name=op.f("ck_users_script")),
        sa.ForeignKeyConstraint(
            ["asha_id"], ["users.id"], name=op.f("fk_users_asha_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("wa_id", name=op.f("uq_users_wa_id")),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("wa_message_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("media_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(direction = 'inbound') = (status IS NOT NULL)",
            name=op.f("ck_messages_only_inbound_has_status"),
        ),
        sa.CheckConstraint(
            "direction IN ('inbound', 'outbound')", name=op.f("ck_messages_direction")
        ),
        sa.CheckConstraint(
            "kind IN ('text', 'audio', 'image', 'template')",
            name=op.f("ck_messages_kind"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name=op.f("ck_messages_status"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_messages_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint("wa_message_id", name=op.f("uq_messages_wa_message_id")),
    )
    op.create_index(
        "ix_messages_pending",
        "messages",
        ["id"],
        unique=False,
        postgresql_where="status = 'pending'",
    )
    op.create_table(
        "cases",
        sa.Column("number", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("cause", sa.Text(), nullable=False),
        sa.Column("patient_id", sa.BigInteger(), nullable=False),
        sa.Column("asha_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), server_default="open", nullable=False),
        sa.Column("asha_reply", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'answered') = "
            "(asha_reply IS NOT NULL AND answered_at IS NOT NULL)",
            name=op.f("ck_cases_answered_has_reply"),
        ),
        sa.CheckConstraint(
            "cause IN ('danger_sign', 'weak_evidence', 'citation_failed', "
            "'critical_lab_value', 'prescription_confirmation', 'missed_doses')",
            name=op.f("ck_cases_cause"),
        ),
        sa.CheckConstraint(
            "status IN ('open', 'answered')", name=op.f("ck_cases_status")
        ),
        sa.ForeignKeyConstraint(
            ["asha_id"], ["users.id"], name=op.f("fk_cases_asha_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"], ["users.id"], name=op.f("fk_cases_patient_id_users")
        ),
        sa.PrimaryKeyConstraint("number", name=op.f("pk_cases")),
    )
    op.create_table(
        "reminders",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("patient_id", sa.BigInteger(), nullable=False),
        sa.Column("medicine", sa.Text(), nullable=False),
        sa.Column("times", sa.ARRAY(sa.Time()), nullable=False),
        sa.Column("starts_on", sa.Date(), nullable=False),
        sa.Column("ends_on", sa.Date(), nullable=True),
        sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "cardinality(times) > 0", name=op.f("ck_reminders_has_times")
        ),
        sa.CheckConstraint(
            "ends_on >= starts_on", name=op.f("ck_reminders_ends_after_start")
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"], ["users.id"], name=op.f("fk_reminders_patient_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reminders")),
    )
    op.create_index(
        "ix_reminders_next_due_at",
        "reminders",
        ["next_due_at"],
        unique=False,
        postgresql_where="next_due_at IS NOT NULL",
    )
    op.create_table(
        "doses",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("reminder_id", sa.BigInteger(), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("taken_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'taken') = (taken_at IS NOT NULL)",
            name=op.f("ck_doses_taken_has_time"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'taken', 'missed')", name=op.f("ck_doses_status")
        ),
        sa.ForeignKeyConstraint(
            ["reminder_id"],
            ["reminders.id"],
            name=op.f("fk_doses_reminder_id_reminders"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_doses")),
        sa.UniqueConstraint("reminder_id", "due_at", name=op.f("uq_doses_reminder_id")),
    )
    op.create_table(
        "kb_chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publisher", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_type = 'health_worker_answer' OR (publisher IS NOT NULL "
            "AND year IS NOT NULL AND page IS NOT NULL AND url IS NOT NULL)",
            name=op.f("ck_kb_chunks_guideline_is_cited"),
        ),
        sa.CheckConstraint(
            "source_type IN ('guideline', 'health_worker_answer')",
            name=op.f("ck_kb_chunks_source_type"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_chunks")),
    )
    op.create_index(
        "ix_kb_chunks_embedding",
        "kb_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table("kb_chunks")
    op.drop_table("doses")
    op.drop_table("reminders")
    op.drop_table("cases")
    op.drop_table("messages")
    op.drop_table("users")
    op.execute("DROP EXTENSION vector")
