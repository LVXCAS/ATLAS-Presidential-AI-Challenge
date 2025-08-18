# LangGraph Trading System - Development Makefile

.PHONY: help setup build up down logs test lint format clean health

# Default target
help:
	@echo "LangGraph Trading System - Development Commands"
	@echo "================================================"
	@echo "setup     - Set up development environment"
	@echo "build     - Build Docker images"
	@echo "up        - Start all services"
	@echo "down      - Stop all services"
	@echo "logs      - View service logs"
	@echo "test      - Run tests"
	@echo "lint      - Run code linting"
	@echo "format    - Format code"
	@echo "clean     - Clean up containers and volumes"
	@echo "health    - Check system health"
	@echo "shell     - Open shell in trading app container"

# Development setup
setup:
	@echo "Setting up development environment..."
	python scripts/setup_dev.py

# Docker commands
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# Development commands
test:
	poetry run pytest

lint:
	poetry run flake8 .
	poetry run mypy .

format:
	poetry run black .
	poetry run isort .

# Cleanup
clean:
	docker compose down -v
	docker system prune -f

# Health check
health:
	python scripts/health_check.py

# Shell access
shell:
	docker compose exec trading_app bash

# Database commands
db-shell:
	docker compose exec postgres psql -U trading_user -d trading_system

redis-shell:
	docker compose exec redis redis-cli -a redis_password

# Install dependencies
install:
	poetry install

# Update dependencies
update:
	poetry update

# Run the application
run:
	python main.py

# Run in Docker
run-docker:
	docker compose up trading_app