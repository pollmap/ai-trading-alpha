.PHONY: install db-up db-down db-init test test-unit test-integration run dashboard lint typecheck clean

install:
	pip install -e ".[dev]"

db-up:
	docker compose up -d

db-down:
	docker compose down

db-init:
	python scripts/setup_db.py

test:
	pytest -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

run:
	python scripts/run_benchmark.py

dashboard:
	streamlit run src/dashboard/app.py

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
