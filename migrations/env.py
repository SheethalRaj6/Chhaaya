from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from chhaaya.db import Base, sqlalchemy_url
from chhaaya.settings import Settings

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

engine = create_engine(sqlalchemy_url(Settings().database_url))
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
