"""Alembic environment.

Migrations always run on a connection handed over by ``researchai.db.migrate``
(``config.attributes["connection"]``); there is no offline SQL mode.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from researchai.db.models import Base

connection: Connection | None = context.config.attributes.get("connection")
if connection is None:
    raise RuntimeError("run migrations via `python -m researchai.db upgrade`")

context.configure(connection=connection, target_metadata=Base.metadata)
with context.begin_transaction():
    context.run_migrations()
