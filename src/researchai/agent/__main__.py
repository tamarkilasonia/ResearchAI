"""Ask one research question from the command line.

    python -m researchai.agent "How many public charging points are there?"
    python -m researchai.agent --json "..."      # full result with trace

Talks to the MCP tool server at MCP_SERVER_URL; LLM_PROVIDER=none runs offline.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from researchai.agent.llm import build_llm
from researchai.agent.models import RunStatus
from researchai.agent.render import render_markdown
from researchai.agent.runner import ResearchAgent
from researchai.config import Settings, get_settings
from researchai.mcp_client.client import ResearchTools


async def ask(question: str, settings: Settings, *, as_json: bool) -> int:
    async with ResearchTools(settings.mcp_server_url, settings.tool_timeout_seconds) as tools:
        agent = ResearchAgent(tools=tools, llm=build_llm(settings), settings=settings)
        result = await agent.run(question)
    print(result.model_dump_json(indent=2) if as_json else render_markdown(result))
    return 0 if result.status in (RunStatus.ANSWERED, RunStatus.ABSTAINED) else 1


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m researchai.agent")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    return asyncio.run(ask(args.question, settings or get_settings(), as_json=args.json))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
