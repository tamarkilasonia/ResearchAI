.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-integration cov mutation check db up down corpus

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv with all dependency groups
	uv sync --frozen

lint: ## ruff lint + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format: ## Apply ruff formatting and safe fixes
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typemutation: ## Plant known bugs; fail if the tests miss one
	uv run python -m scripts.mutation_check

check: ## mypy --strict
	uv run mypy

test: ## Unit tests (random order, no network)
	uv run pytest

test-integration: ## Integration tests (needs `make db`)
	uv run pytest -m integration

cov: ## Unit + integration tests with the 95% coverage gate (needs `make db`)
	uv run pytest -m "integration or not integration" --cov --cov-report=term-missing

check: lint typecheck cov ## Everything CI runs before the docker build

corpus: ## Validate the corpus and check the frozen manifest
	uv run python -m researchai.corpus validate
	uv run python -m researchai.corpus manifest --check

db: ## Start only Postgres + pgvector (for integration tests)
	docker compose up -d --wait db

up: ## Start the local stack
	docker compose up -d --build

down: ## Stop the local stack
	docker compose down
