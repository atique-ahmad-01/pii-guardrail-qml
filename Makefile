PROFILE ?= smoke
CFG := configs/$(PROFILE).yaml
UV ?= $(shell command -v uv 2>/dev/null || echo .tools/uv)
RUN := $(UV) run python

.PHONY: setup test lint fmt data stress gate masker pipeline report smoke reproduce clean

setup:
	$(UV) sync --all-extras

test:
	$(UV) run pytest -m "not slow"

test-all:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

fmt:
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

data:
	$(RUN) scripts/prepare_data.py --config $(CFG)

stress:
	$(RUN) scripts/run_stress.py --config $(CFG)

gate:
	$(RUN) scripts/run_gate.py --config $(CFG)

masker:
	$(RUN) scripts/run_masker.py --config $(CFG)

pipeline:
	$(RUN) scripts/run_pipeline.py --config $(CFG)

report:
	$(RUN) scripts/make_report.py --config $(CFG)

# Everything on the tiny profile (<15 min on a laptop).
smoke:
	$(MAKE) PROFILE=smoke data stress gate masker pipeline report

# Full data -> report. Long: run on a CUDA GPU.
reproduce:
	$(MAKE) PROFILE=full data stress gate masker pipeline report

clean:
	rm -rf .pytest_cache .ruff_cache build dist
