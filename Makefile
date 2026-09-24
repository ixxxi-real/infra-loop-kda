.PHONY: help setup init check-python venv install validate test task-test bench-test \
	precision-test control-test flow-test overlay-test hook-test packaging-test \
	all-tests workload-example status task-list doctor doctor-agent verify package \
	repository-check install-check \
	overlay-build overlay-verify workspace-prepare workspace-status agent-plan \
	agent-status run-plan export-bundle skills-check skills-install clean-venv

CONFIG ?= config/project.example.json
PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON ?= $(VENV)/bin/python
K3CTL ?= $(PYTHON) tools/k3ctl.py
TASK ?=
TASK_ARG = $(if $(TASK),--task $(TASK),)

help:
	@echo "Setup"
	@echo "  make setup              bootstrap: submodules, venv, install, validate, test"
	@echo "  make init               initialize pinned submodules only"
	@echo "  make install            install the control-plane wheel into .venv"
	@echo ""
	@echo "Inspection (read-only, offline)"
	@echo "  make doctor             toolchain report; works without submodules"
	@echo "  make doctor-agent       also check bash>=4, jq, claude, codex"
	@echo "  make validate           validate project metadata"
	@echo "  make status             project and task status"
	@echo "  make task-list          list configured tasks"
	@echo ""
	@echo "Task flow (TASK=<id> selects a task; defaults to active_task)"
	@echo "  make workspace-prepare  materialise a task's exact base commit"
	@echo "  make workspace-status   report prepared workspace state"
	@echo "  make agent-plan         dry-run the Claude+Humanize loop; spawns nothing"
	@echo "  make agent-status       real loop/process state"
	@echo "  make run-plan           freeze a runner plan; starts nothing"
	@echo "  make export-bundle      write a fresh evidence bundle"
	@echo ""
	@echo "No-commit overlay"
	@echo "  make overlay-build      build the audited Humanize compat overlay"
	@echo "  make overlay-verify     re-hash and verify the built overlay"
	@echo ""
	@echo "Tests"
	@echo "  make test               control-plane tests"
	@echo "  make task-test          prefill bench + precision suites"
	@echo "  make all-tests          everything below plus task-test"
	@echo "  make verify             repository checks, validation and tests"
	@echo "  make package            build wheel + source archive into dist/"
	@echo "  make install-check      install from a wheel in a temporary venv"
	@echo "  make control-test flow-test overlay-test hook-test packaging-test"
	@echo ""
	@echo "Skills"
	@echo "  make skills-check       report what would be linked; writes nothing"
	@echo "  make skills-install     link skills (see scripts/install-skills.sh --help)"

# A full environment bootstrap is intentionally explicit. It fetches only the
# pinned source/workflow repositories and installs this dependency-free control
# plane; CUDA/PyTorch/SGLang runtime installation remains machine-specific.
setup: check-python init venv install validate test

check-python:
	$(PYTHON) -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9 or newer is required"'

init:
	git submodule sync --recursive
	git submodule update --init --recursive

venv:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	@if ! $(VENV_PYTHON) -c 'import pip' >/dev/null 2>&1; then \
		echo "venv: pip is missing; bootstrapping it with ensurepip"; \
		$(VENV_PYTHON) -m ensurepip --upgrade; \
	fi

# Install a wheel rather than invoking the deprecated `setup.py develop` path.
# This also works with the older pip bundled by some Python 3.9 distributions
# and keeps the installed console script independent of the caller's cwd.
#
# NETWORK: by default pip builds in an isolated environment, which FETCHES the
# build backend (setuptools>=64, per pyproject.toml) from a package index. That
# is the only network access this target needs; the project itself has no
# runtime dependencies.
#
# OFFLINE: possible when the backend is already importable in the virtualenv:
#   make install PIP_INSTALL_FLAGS="--no-build-isolation --no-index"
PIP_INSTALL_FLAGS ?=
install: venv
	@mkdir -p .infra/build
	@find .infra/build -maxdepth 1 -name 'infra_loop_kda-*.whl' -delete
	$(VENV_PYTHON) -m pip wheel $(PIP_INSTALL_FLAGS) --no-deps --wheel-dir .infra/build .
	$(VENV_PYTHON) -m pip install --no-deps --force-reinstall .infra/build/infra_loop_kda-*.whl

# ------------------------------------------------------------------ inspection

doctor:
	./scripts/doctor.sh --config $(CONFIG) $(TASK_ARG)

doctor-agent:
	./scripts/doctor.sh --config $(CONFIG) $(TASK_ARG) --agent-profile

validate:
	$(K3CTL) validate --config $(CONFIG)

status:
	$(K3CTL) status --config $(CONFIG)

task-list:
	$(K3CTL) task-list --config $(CONFIG)

# ------------------------------------------------------------------ task flow

workspace-prepare:
	$(K3CTL) workspace prepare --config $(CONFIG) $(TASK_ARG)

workspace-status:
	$(K3CTL) workspace status --config $(CONFIG) $(TASK_ARG)

agent-plan:
	$(K3CTL) agent plan --config $(CONFIG) $(TASK_ARG)

agent-status:
	$(K3CTL) agent status --config $(CONFIG) $(TASK_ARG)

run-plan:
	$(K3CTL) run plan --config $(CONFIG) $(TASK_ARG)

export-bundle:
	$(K3CTL) export bundle --config $(CONFIG) $(TASK_ARG)

# --------------------------------------------------------------------- overlay

overlay-build:
	$(K3CTL) overlay build --config $(CONFIG)

overlay-verify:
	$(K3CTL) overlay verify --config $(CONFIG)

# ----------------------------------------------------------------------- tests

test:
	$(PYTHON) -m unittest discover -s tests -v

control-test:
	$(PYTHON) -m unittest tests.test_control -v

flow-test:
	$(PYTHON) -m unittest tests.test_flow -v

overlay-test:
	$(PYTHON) -m unittest tests.test_overlay_hooks -v

hook-test:
	$(PYTHON) -m unittest tests.test_hook_chain -v

packaging-test:
	$(PYTHON) -m unittest tests.test_packaging -v

task-test: bench-test precision-test

bench-test:
	$(PYTHON) -B -m unittest discover -s projects/kimi-k3/prefill/bench -p 'test_*.py' -v

precision-test:
	$(PYTHON) -B -m unittest discover -s projects/kimi-k3/prefill/precision -p 'test_*.py' -v

all-tests: test task-test

repository-check:
	./scripts/verify-repository.sh

verify: repository-check validate all-tests workload-example

# Build without requiring the `build` package or a network connection after
# `make setup` has created the virtual environment. The isolated source tree
# still uses the PEP 517 metadata from pyproject.toml.
package: install
	$(VENV_PYTHON) -m pip wheel . --no-deps --wheel-dir dist
	$(VENV_PYTHON) setup.py sdist --dist-dir dist

install-check: package
	@tmp_dir=$$(mktemp -d); \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	$(PYTHON) -m venv "$$tmp_dir/venv"; \
	"$$tmp_dir/venv/bin/python" -m pip install --no-deps dist/infra_loop_kda-*.whl >/dev/null; \
	cd "$$tmp_dir"; \
	"$$tmp_dir/venv/bin/k3ctl" --help >/dev/null; \
	K3_PROJECT_ROOT="$(CURDIR)" "$$tmp_dir/venv/bin/k3ctl" validate --config config/project.example.json

workload-example:
	$(PYTHON) projects/kimi-k3/prefill/scripts/configure_workloads.py \
		--checkpoint-config projects/kimi-k3/prefill/checkpoint-config.example.json \
		--deployment projects/kimi-k3/prefill/deployment.example.json \
		--output /tmp/kimi-k3-workloads.json

# ---------------------------------------------------------------------- skills

# Reports what would be linked and writes nothing.
skills-check:
	./scripts/install-skills.sh

# Writes symlinks into the codex-bak home by default. Set CODEX_HOME, or pass an explicit
# destination, to keep it isolated:
#   make skills-install SKILLS_ARGS="--home $(PWD)/.codex-test"
SKILLS_ARGS ?= --apply
skills-install:
	./scripts/install-skills.sh $(SKILLS_ARGS)

clean-venv:
	rm -rf $(VENV)
