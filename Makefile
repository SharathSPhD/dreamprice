.PHONY: build shell test train serve lint format check clean help

IMAGE   := dreamprice:latest
SERVICE := dreamprice
RUN     := docker compose run --rm $(SERVICE)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker compose build

shell: ## Interactive bash inside the container with GPU
	docker compose run --rm --service-ports $(SERVICE) bash

test: ## Run pytest inside container
	$(RUN) pytest tests/ -v --tb=short

test-collect: ## Collect tests without running (verify imports)
	$(RUN) pytest tests/ --co -q

train: ## Run training (pass ARGS for Hydra overrides, e.g. make train ARGS="experiment=main")
	$(RUN) python scripts/train.py $(ARGS)

serve: ## Start FastAPI server on port 8000
	docker compose run --rm --service-ports $(SERVICE) \
		python -m retail_world_model.api.serve

lint: ## Run ruff check + pyright
	$(RUN) sh -c "ruff check src/ tests/ && pyright src/"

format: ## Run ruff format
	$(RUN) ruff format src/ tests/

check: ## Full quality gate: lint + test
	$(RUN) sh -c "ruff check src/ tests/ && ruff format --check src/ tests/ && pyright src/ && pytest tests/ -v --tb=short"

verify-gpu: ## Verify CUDA + mamba-ssm inside container
	$(RUN) python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; from mamba_ssm import Mamba2; print('mamba-ssm: OK'); import retail_world_model; print(f'retail_world_model: OK'); print('All checks passed.')"

clean: ## Remove built images and caches
	docker compose down --rmi local --volumes --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
