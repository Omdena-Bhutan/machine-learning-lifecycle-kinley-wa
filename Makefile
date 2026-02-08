.PHONY: help install download preprocess train evaluate test api docker clean format lint

help:
	@echo "ML Lifecycle Project - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install all dependencies"
	@echo "  make download      - Download and prepare dataset"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make preprocess    - Preprocess data and create train/test splits"
	@echo "  make train         - Train the sentiment analysis model"
	@echo "  make evaluate      - Evaluate model on test set"
	@echo "  make pipeline      - Run complete pipeline (download -> preprocess -> train -> evaluate)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          - Run unit tests with pytest"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo ""
	@echo "API & Deployment:"
	@echo "  make api           - Start Flask API server"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo ""
	@echo "Utility:"
	@echo "  make mlflow-ui     - Start MLflow UI"
	@echo "  make clean         - Clean up generated files"
	@echo ""

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed successfully"

download:
	python src/download_data.py
	@echo "✓ Dataset downloaded and prepared"

preprocess:
	python src/data_loader.py
	@echo "✓ Data preprocessing completed"

train:
	python src/model.py
	@echo "✓ Model training completed"

evaluate:
	python src/evaluate.py
	@echo "✓ Model evaluation completed"

pipeline: download preprocess train evaluate
	@echo "✓ Complete ML pipeline executed successfully"

test:
	pytest tests/ -v --tb=short
	@echo "✓ All tests passed"

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "✓ Test coverage report generated"

lint:
	flake8 src/ tests/ app/ --max-line-length=120 --ignore=E203,W503 || true
	@echo "✓ Linting completed"

format:
	black src/ tests/ app/ --line-length=120
	isort src/ tests/ app/
	@echo "✓ Code formatted successfully"

api:
	python app/api.py
	@echo "API server started on http://localhost:5000"

docker-build:
	cd app && docker build -t sentiment-api:latest .
	@echo "✓ Docker image built: sentiment-api:latest"

docker-run:
	docker run -p 5000:5000 --name sentiment-api sentiment-api:latest
	@echo "✓ Docker container running on http://localhost:5000"

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000
	@echo "MLflow UI started on http://localhost:5000"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	rm -rf *.egg-info dist build
	@echo "✓ Cleanup completed"
