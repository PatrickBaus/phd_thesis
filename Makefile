# Fail loudly: strict shell, no silent undefined variables, no leftover
# half-written files, no implicit rules we don't use.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.NOTPARALLEL:
.DEFAULT_GOAL := all
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules --no-builtin-variables

SOURCES=main.tex
PDF_OBJECTS=$(SOURCES:.tex=.pdf)

LATEXMK=latexmk
LATEXMK_OPTIONS=-lualatex
# Extra latexmk options, e.g. `make tex LATEXMK_OPTIONS_EXTRA=-silent`.
# Exported so it reaches the container through `-e` without any shell quoting.
LATEXMK_OPTIONS_EXTRA ?=
export LATEXMK_OPTIONS_EXTRA

DOCKER=docker
# Pin by digest for fully reproducible builds, e.g.
#   texlive/texlive:TL2023-historic@sha256:<digest>
DOCKER_CONTAINER=texlive/texlive:TL2023-historic
# The TeX sources are code (LuaLaTeX can run Lua), so build in a locked-down
# container: no network, no capabilities, no privilege escalation, and as the
# calling user so generated files are not owned by root.
# Set DOCKER_NETWORK=bridge if a build genuinely needs network access.
DOCKER_NETWORK ?= none
DOCKER_RUN = $(DOCKER) run --rm \
  --network $(DOCKER_NETWORK) \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --user "$$(id -u):$$(id -g)" \
  -e HOME=/tmp \
  -e LATEXMK_OPTIONS_EXTRA \
  -v "$(CURDIR):/tex" \
  -w /tex

.PHONY: all pdf tex figures clean dist-clean debug FORCE

all: figures tex

pdf: $(PDF_OBJECTS)

# latexmk tracks the real dependencies (\input, figures, ...) itself, so always
# call it; make's timestamp check on the .tex file alone would miss changes.
%.pdf: %.tex FORCE
	@echo Input file: $<
	$(LATEXMK) $(LATEXMK_OPTIONS_EXTRA) $(LATEXMK_OPTIONS) $<

FORCE:

clean:
	-$(LATEXMK) -C $(SOURCES:.tex=)

dist-clean: clean
	$(MAKE) -C data clean

tex:
	$(DOCKER_RUN) $(DOCKER_CONTAINER) make pdf

figures:
	$(MAKE) -C data

debug:
	$(DOCKER_RUN) -it $(DOCKER_CONTAINER) bash
