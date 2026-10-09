"""Database command line.

python -m researchai.db upgrade [revision]     default: head
python -m researchai.db downgrade [revision]   default: base
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from researchai.config import Settings, get_settings
from researchai.db import migrate
from researchai.db.session import create_engine


async def _run(settings: Settings, command: str, revision: str) -> None:
    engine = create_engine(settings)
    try:
        if command == "upgrade":
            await migrate.upgrade(engine, embedding_dim=settings.embedding_dim, revision=revision)
        else:
            await migrate.downgrade(engine, embedding_dim=settings.embedding_dim, revision=revision)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m researchai.db")
    parser.add_argument("command", choices=["upgrade", "downgrade"])
    parser.add_argument("revision", nargs="?")
    args = parser.parse_args(argv)
    revision = args.revision or ("head" if args.command == "upgrade" else "base")
    asyncio.run(_run(settings or get_settings(), args.command, revision))
    print(f"{args.command} -> {revision}: ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
