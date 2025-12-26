.PHONY: help install dev-install docker-build docker-run train predict submit test lint format clean data-download

# Variables
PROJECT_NAME := ecg-digitization
DOCKER_IMAGE := ecg-digitization:latest
DOCKER_BASE := trading_base:latest
DATA_DIR := data
MODEL_DIR := models

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in production mode"
	@echo "  make dev-install   - Install package with dev dependencies"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container interactively"
	@echo "  make docker-train  - Run training inside Docker"
	@echo "  make docker-shell  - Open shell in Docker container"
	@echo "  make train         - Run training pipeline"
	@echo "  make predict       - Run inference pipeline"
	@echo "  make submit        - Generate submission file"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make data-setup    - Setup data directories"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev,jupyter]"

# Docker commands
docker-build:
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-e HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) \
		$(DOCKER_IMAGE)

docker-train:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-e HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) \
		$(DOCKER_IMAGE) python -m ecg_digitization.train

docker-shell:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-e HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) \
		$(DOCKER_IMAGE) /bin/bash

docker-compose-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-compose-down:
	docker-compose -f docker/docker-compose.yml down

# Training and inference
train:
	python -m ecg_digitization.train

predict:
	python -m ecg_digitization.predict

submit:
	python -m ecg_digitization.submit

validate:
	python -m ecg_digitization.validate

# Ray-based training (distributed/parallel)
train-ray:
	python -m ecg_digitization.train_ray mode=train

# Ray hyperparameter tuning
tune:
	python -m ecg_digitization.train_ray mode=tune

# Docker Ray training
docker-train-ray:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-e HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5050 \
		--shm-size=8g \
		$(DOCKER_IMAGE) python -m ecg_digitization.train_ray mode=train

# Docker Ray hyperparameter tuning
docker-tune:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-e HUGGING_FACE_HUB_TOKEN=$(HUGGING_FACE_HUB_TOKEN) \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5050 \
		--shm-size=8g \
		$(DOCKER_IMAGE) python -m ecg_digitization.train_ray mode=tune

# Development
test:
	pytest tests/ -v --cov=src/ecg_digitization --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# Data
data-setup:
	mkdir -p $(DATA_DIR)/raw
	mkdir -p $(DATA_DIR)/processed
	mkdir -p $(DATA_DIR)/train
	mkdir -p $(DATA_DIR)/test
	mkdir -p $(DATA_DIR)/submissions
	mkdir -p $(MODEL_DIR)/checkpoints
	mkdir -p $(MODEL_DIR)/exports
	touch $(DATA_DIR)/raw/.gitkeep
	touch $(DATA_DIR)/processed/.gitkeep
	touch $(DATA_DIR)/train/.gitkeep
	touch $(DATA_DIR)/test/.gitkeep
	touch $(DATA_DIR)/submissions/.gitkeep
	touch $(MODEL_DIR)/checkpoints/.gitkeep
	touch $(MODEL_DIR)/exports/.gitkeep

# Kaggle data download (requires kaggle CLI)
kaggle-download:
	kaggle competitions download -c physionet-ecg-image-digitization -p $(DATA_DIR)/raw
	unzip $(DATA_DIR)/raw/physionet-ecg-image-digitization.zip -d $(DATA_DIR)/raw

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-models:
	rm -rf $(MODEL_DIR)/checkpoints/*
	rm -rf $(MODEL_DIR)/exports/*
	touch $(MODEL_DIR)/checkpoints/.gitkeep
	touch $(MODEL_DIR)/exports/.gitkeep

clean-data:
	rm -rf $(DATA_DIR)/processed/*
	touch $(DATA_DIR)/processed/.gitkeep

# EDA and visualization
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# MLflow
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000
