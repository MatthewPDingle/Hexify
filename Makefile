.PHONY: install test lint format clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check v2/
	mypy v2/ --ignore-missing-imports

format:
	ruff format v2/
	ruff check --fix v2/

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
