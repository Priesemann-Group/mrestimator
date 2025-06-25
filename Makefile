SHELL := /bin/bash

.DEFAULT_GOAL := help

PACKAGE_PATH="mrestimator"

SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = mrestimator
SOURCEDIR     = docs/
BUILDDIR      = docs/_build

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: ## remove build artifacts, compiled files, and cache
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {}  +
	find . -name '*~' -exec rm -f {} +
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY: clean-docs
clean-docs: ## remove documentation build artifacts
	rm -fr docs/_build/

.PHONY: lint
lint: ## lint the project
	pre-commit run --all-files

.PHONY: test
test: ## run tests quickly with the default Python
	pytest

.PHONY: install
install: ## install the package in development mode
	pip install -e .

.PHONY: install-dev
install-dev: ## install the package with development dependencies
	pip install -e .[dev]

.PHONY: install-docs
install-docs: ## install the package with documentation dependencies
	pip install -e .[docs]

.PHONY: docs-build
docs-build: ## build documentation
	sphinx-apidoc -o docs/_build ${PACKAGE_PATH}
	$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -P

.PHONY: docs-clean
docs-clean: clean-docs docs-build ## clean and rebuild documentation

.PHONY: docs-preview
docs-preview: docs-build ## build documentation and serve it locally
	cd docs/_build && python -m http.server

.PHONY: docs-html
docs-html: ## build HTML documentation (alias for docs-build)
	$(MAKE) -C docs html

.PHONY: build
build: ## build distribution packages
	python -m build

.PHONY: format
format: ## format code with ruff
	ruff format .
	ruff check --fix .

.PHONY: check
check: ## run all checks (lint, test)
	$(MAKE) lint
	$(MAKE) test

.PHONY: all
all: clean install-dev format check docs-build ## run full development cycle
