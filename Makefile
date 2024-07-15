SOURCES=main.tex
PDF_OBJECTS=$(SOURCES:.tex=.pdf)

LATEXMK=latexmk
LATEXMK_OPTIONS=-lualatex

DOCKER=docker
DOCKER_COMMAND=run --rm -w /tex/ --env LATEXMK_OPTIONS_EXTRA=$(LATEXMK_OPTIONS_EXTRA)
DOCKER_MOUNT=-v`pwd`:/tex
DOCKER_CONTAINER=texlive/texlive:TL2023-historic

all: figures tex

pdf: $(PDF_OBJECTS)

%.pdf: %.tex
	@echo Input file: $<
	$(LATEXMK) $(LATEXMK_OPTIONS_EXTRA) $(LATEXMK_OPTIONS) $<

clean:
	-$(LATEXMK) -C main

dist-clean: clean
	-rm $(FILENAME).tar.gz
	@cd data && $(MAKE) clean

.PHONY: tex
tex:
	$(DOCKER) $(DOCKER_COMMAND) $(DOCKER_MOUNT) $(DOCKER_CONTAINER) \
		make pdf

.PHONY: figures
figures:
	cd data && $(MAKE)

debug:
	$(DOCKER) $(DOCKER_COMMAND) -it $(DOCKER_MOUNT) $(DOCKER_CONTAINER) \
		bash
