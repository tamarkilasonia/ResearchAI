.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration cov check up down

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv with all dependency groups
	uv sync --frozen

lint: ## ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

format: ## Apply ruff formatting and safe fixes
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## mypy --strict
	uv run mypy

test: ## Unit tests (random order, no network)
	uv run pytest

test-integration: ## Integration tests (needs `make up`)
	uv run pytest -m integration

cov: ## Unit tests with the 95% coverage gate
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck cov ## Everything CI runs before integration

up: ## Start the local stack
	docker compose up -d --build

down: ## Stop the local stack
	docker compose down
