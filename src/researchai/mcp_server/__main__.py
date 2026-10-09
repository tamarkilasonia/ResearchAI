"""Run the MCP tool server: ``python -m researchai.mcp_server``."""

from __future__ import annotations

import asyncio

import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from researchai.config import Settings, get_settings
from researchai.corpus.embeddings import build_embedder
from researchai.db.session import create_engine, session_factory
from researchai.log import configure_logging
from researchai.mcp_server.server import ToolServices, build_server
from researchai.tools.documents import DocumentStore
from researchai.tools.search import CorpusSearch
from researchai.tools.web import LiveWebSearch


def build_app(settings: Settings, services: ToolServices) -> Starlette:
    return build_server(services).streamable_http_app(
        stateless_http=True,
        json_response=True,
        host=settings.mcp_host,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.mcp_allowed_host_list,
        ),
    )


async def serve(settings: Settings) -> None:  # pragma: no cover - exercised by docker compose
    configure_logging(settings)
    engine = create_engine(settings)
    sessions = session_factory(engine)
    services = ToolServices(
        search=CorpusSearch(sessions, build_embedder(settings)),
        documents=DocumentStore(sessions),
        web=LiveWebSearch(settings.live_web_search_url)
        if settings.live_web_search_enabled
        else None,
    )
    config = uvicorn.Config(
        build_app(settings, services),
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_config=None,
    )
    try:
        await uvicorn.Server(config).serve()
    finally:
        await engine.dispose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(serve(get_settings()))
