SOURCES=main.tex
PDF_OBJECTS=$(SOURCES:.tex=.pdf)

LATEXMK=latexmk
LATEXMK_OPTIONS=-lualatex

DOCKER=docker
DOCKER_COMMAND=run --rm -w /tex/ --env LATEXMK_OPTIONS_EXTRA=$(LATEXMK_OPTIONS_EXTRA)
DOCKER_MOUNT=-v`pwd`:/tex
export DOCKER_CONTAINER=texlive/texlive:TL2022-historic

all: tex figures

pdf: $(PDF_OBJECTS)

%.pdf: %.tex
	@echo Input file: $<
	$(LATEXMK) $(LATEXMK_OPTIONS_EXTRA) $(LATEXMK_OPTIONS) $<

clean:
	-$(LATEXMK) -C main
	-make -C figures clean

dist-clean: clean
	-rm $(FILENAME).tar.gz

.PHONY: tex
tex:
	$(DOCKER) $(DOCKER_COMMAND) $(DOCKER_MOUNT) texlive/texlive:TL2022-historic \
		make pdf

.PHONY: figures
figures:
	make -C data

debug:
	$(DOCKER) $(DOCKER_COMMAND) -it $(DOCKER_MOUNT) texlive/texlive:TL2022-historic \
		bash
