.PHONY: format lint test clean

format:
	black .
	isort .

lint:
	flake8 .
	mypy .
	black --check .
	isort --check .

test:
	pytest tests/ --cov=hvit --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
