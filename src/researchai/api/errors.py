"""One error format for every failure: ``{"error": {"code", "message", "request_id"}}``.

Stack traces and exception details never reach the client; unexpected errors
are logged server-side with the request id that the client receives.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from researchai.log import bind_request_id, current_request_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

HTTP_CODES = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "INVALID_REQUEST",
    429: "TOO_MANY_REQUESTS",
    503: "SERVICE_UNAVAILABLE",
}


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ApiError(Exception):
    """An anticipated API failure with a stable code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=current_request_id())
    )
    return JSONResponse(status_code=status, content=body.model_dump())


async def _api_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, ApiError)  # noqa: S101 - registered for ApiError only
    return error_response(exc.status, exc.code, exc.message)


async def _http_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    code = HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
    message = exc.detail if isinstance(exc.detail, str) else code.replace("_", " ").lower()
    return error_response(exc.status_code, code, message)


async def _validation_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    problems = [
        f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
    ]
    return error_response(422, "INVALID_REQUEST", "; ".join(problems)[:500])


def install(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    # Unexpected exceptions are caught by the middleware below, which also keeps
    # the request id on the 500 response.

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        rid = supplied if _VALID_REQUEST_ID.match(supplied) else uuid.uuid4().hex
        with bind_request_id(rid):
            try:
                response = await call_next(request)
            except Exception:  # last line of defence: nothing may leak
                logger.exception("unhandled error", extra={"path": request.url.path})
                response = error_response(500, "INTERNAL", "internal error")
        response.headers[REQUEST_ID_HEADER] = rid
        return response
