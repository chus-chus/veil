.PHONY: help lint format test compare viz

.DEFAULT_GOAL := help

lint/black: ## check style with black
	black --check veil

lint/isort: ## check style with isort
	isort --check-only --profile black veil

lint/autoflake: ## check for unused imports
	autoflake --recursive --remove-all-unused-imports --check veil

lint/pyright: ## run type checking
	pyright

lint/codespell:
	codespell --skip './env/**,./docs/_build/**' -L inout

lint: lint/isort lint/black lint/autoflake lint/codespell lint/pyright	## check style

format/black: ## format code with black
	black veil

format/isort: ## format code with isort
	isort --profile black veil

format/autoflake: ## remove unused imports
	autoflake --in-place --recursive --remove-all-unused-imports veil

format: format/isort format/autoflake format/black ## format code

test/e2e:
	python3 -m pytest -m e2e -s

test/unit:
	python3 -m pytest -m unit -s

test/performance:
	python3 -m pytest -m performance -s

# e2e and performance tests will need a server running locally, by default on port 8000
test/all: test/e2e test/unit test/performance

compare:
	@# Prefer explicit runs if provided via variables or positional args
	ALLARGS=$(filter-out compare,$(MAKECMDGOALS)); \
	if [ -n "$(BASELINE)$(CANDIDATES)$(RUNS)" ]; then \
		EXPLICIT_RUNS="$(if $(BASELINE),$(BASELINE) )$(if $(CANDIDATES),$(CANDIDATES),$(RUNS))"; \
		set -- $$EXPLICIT_RUNS; \
		echo "Comparing runs: $$*"; \
		python3 -m veil.tools.compare_runs --open "$$@"; \
		exit $$?; \
	fi; \
	# If positional args look like run dirs, treat them as explicit runs
	if [ -n "$$ALLARGS" ]; then \
		POS_RUNS=""; \
		for a in $$ALLARGS; do \
			if [ -d "$$a" ] && [ -f "$$a/metrics.json" ]; then POS_RUNS="$$POS_RUNS $$a"; fi; \
		done; \
		if [ -n "$$POS_RUNS" ]; then \
			set -- $$POS_RUNS; \
			echo "Comparing runs: $$*"; \
			python3 -m veil.tools.compare_runs --open "$$@"; \
			exit $$?; \
		fi; \
	fi; \
	# Otherwise, treat first arg or RUN_DIR as a base directory and auto-select runs
	BASE_DIR=$(if $(RUN_DIR),$(RUN_DIR),$(firstword $$ALLARGS)); \
	if [ -z "$$BASE_DIR" ]; then BASE_DIR="veil_runs/eval_set"; fi; \
	if [ ! -d "$$BASE_DIR" ]; then echo "Directory not found: $$BASE_DIR"; exit 1; fi; \
	LATEST_RUN=$$(ls -td "$$BASE_DIR"/run-* 2>/dev/null | head -1); \
	if [ -z "$$LATEST_RUN" ]; then echo "No run directories found under $$BASE_DIR"; exit 1; fi; \
	LATEST_DAY=$$(basename "$$LATEST_RUN" | sed -E 's/^run-([0-9]{8}).*/\1/'); \
	RUNS_LIST=""; \
	for p in $$(ls -td "$$BASE_DIR"/run-* 2>/dev/null); do \
		name=$$(basename "$$p"); \
		day=$$(echo "$$name" | sed -E 's/^run-([0-9]{8}).*/\1/'); \
		if [ "$$day" = "$$LATEST_DAY" ]; then RUNS_LIST="$$RUNS_LIST $$p"; fi; \
	done; \
	NUM_RUNS=$$(echo "$$RUNS_LIST" | wc -w); \
	if [ "$$NUM_RUNS" -lt 2 ]; then \
		RUNS_LIST=$$(ls -td "$$BASE_DIR"/run-* 2>/dev/null | head -2); \
		NUM_RUNS=$$(echo "$$RUNS_LIST" | wc -w); \
	fi; \
	if [ "$$NUM_RUNS" -lt 2 ]; then echo "Need at least two runs under $$BASE_DIR to compare"; exit 1; fi; \
	echo "Comparing runs: $$RUNS_LIST"; \
	set -- $$RUNS_LIST; \
	python3 -m veil.tools.compare_runs --open "$$@"

# Allow: make viz or make viz <output_dir> or make viz RUN_DIR=<output_dir>
RUN_DIR ?=
# Remove the target name 'viz' from goals to get positional arg if provided
ARGS := $(filter-out viz,$(MAKECMDGOALS))
viz:
	@BASE_DIR=$(if $(RUN_DIR),$(RUN_DIR),$(firstword $(ARGS))); \
	if [ -z "$$BASE_DIR" ]; then BASE_DIR="veil_runs/eval_set"; fi; \
	if [ ! -d "$$BASE_DIR" ]; then echo "Directory not found: $$BASE_DIR"; exit 1; fi; \
	RUN_DIR_PATH=$$(ls -td $$BASE_DIR/run-* 2>/dev/null | head -1); \
	if [ -z "$$RUN_DIR_PATH" ]; then echo "No run directories found under $$BASE_DIR"; exit 1; fi; \
	echo "Visualizing latest run: $$RUN_DIR_PATH"; \
	python3 -m veil.tools.visualize_processed --processed-dir "$$RUN_DIR_PATH" --open

# Swallow extra args when using viz/compare so make doesn't try to build them
ifeq (,$(filter compare viz,$(MAKECMDGOALS)))
# no-op when neither compare nor viz is called
else
EXTRA_ARGS := $(filter-out compare viz,$(MAKECMDGOALS))
.PHONY: $(EXTRA_ARGS)
$(EXTRA_ARGS):
	@:
endif

.PHONY: build
build: ## Create mamba environment and set up dependencies
	@echo "Creating mamba environment in ./env..."
	mamba create -p ./env python=3.12 -y
	@echo "Activating environment and running setup..."
	mamba run -p ./env bash -c "cd $(shell pwd) && python3 -m pip install uv"
	mamba run -p ./env bash -c "cd $(shell pwd) && python3 -m uv pip install -r requirements.txt"
	mamba run -p ./env bash -c "cd $(shell pwd) && python3 -m uv pip install -e ."
	@echo "Environment setup complete. To activate the environment, run:"
	@echo "  mamba activate ./env"

.PHONY: docker/build
docker/build: ## Build the GPU Docker image
	DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 -f Dockerfile.gpu -t username/veil:gpu-latest --load .

.PHONY: docs/html docs/clean docs/serve
# Prefer project venv Sphinx; allow override: make docs/html SPHINXBUILD=... 
SPHINXBUILD ?= ./env/bin/sphinx-build
docs/html: ## Build Sphinx HTML docs
	$(SPHINXBUILD) -b html docs docs/_build/html

docs/clean: ## Clean built docs
	rm -rf docs/_build

docs/serve: ## Serve built docs locally at http://localhost:5500
	python3 -m http.server --directory docs/_build/html 5500