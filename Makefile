#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = music_interpret
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python
MMT_DIR = test
MMT_REPO = https://github.com/salu133445/mmt.git

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Initialize the project set up
.PHONY: init
init: clone-mmt create_environment
	@echo "Project initialized."

## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Clone base mmt project
.PHONY: clone-mmt
clone-mmt:
	@if [ -d "$(MMT_DIR)" ]; then \
		echo "MMT repo already exists at $(MMT_DIR)"; \
	else \
		echo "Cloning MMT repository into $(MMT_DIR)..."; \
		git clone $(MMT_REPO) $(MMT_DIR); \
	fi

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
