.PHONY: install run debug clean lint lint-strict test docker-build docker-run docker-interactive docker-help help check-capacity

PYTHON ?= uv run python3

# ============================================================================
# Development Targets
# ============================================================================

install:
	uv sync

run:
	$(PYTHON) -m student --help

help:
	$(PYTHON) -m student help

debug:
	$(PYTHON) -m pdb -m student --help

clean:
	@echo "Cleaning caches..."
	@python3 -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@python3 -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.mypy_cache', '.pytest_cache', '.ruff_cache', 'build', 'dist']]"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q

# ============================================================================
# Docker Targets
# ============================================================================

check-capacity:
	$(PYTHON) check_capacity.py

docker-build:
	docker build -t ragagainsthemachine:latest .

docker-help:
	docker run --rm ragagainsthemachine:latest help

docker-run:
	docker-compose run --rm rag

docker-interactive:
	docker-compose run -it --rm rag /bin/bash

docker-index:
	docker-compose run --rm rag index

docker-search:
	@read -p "Enter search query: " query; \
	docker-compose run --rm rag search "$$query" --k 10

docker-answer:
	@read -p "Enter question: " question; \
	docker-compose run --rm rag answer "$$question" --k 10

docker-clean:
	docker system prune -f
	docker image prune -f

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: help-all
help-all:
	@echo "RAG Against the Machine - Make Targets"
	@echo ""
	@echo "Development:"
	@echo "  make install          - Install dependencies (uv sync)"
	@echo "  make run              - Show help menu"
	@echo "  make help             - Show help with examples"
	@echo "  make lint             - Check code style (flake8 + mypy)"
	@echo "  make lint-strict      - Strict type checking"
	@echo "  make test             - Run pytest"
	@echo "  make clean            - Clean caches and temp files"
	@echo ""
	@echo "Docker/Deployment:"
	@echo "  make check-capacity   - Verify system meets requirements"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-help      - Show help in container"
	@echo "  make docker-index     - Build index in container"
	@echo "  make docker-interactive - Interactive bash shell in container"
	@echo "  make docker-clean     - Clean up Docker images/containers"
	@echo ""
	@echo "See README.md and DEPLOYMENT.md for more information."

