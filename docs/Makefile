PDF = dissertation.pdf
TEX = dissertation.tex

all: $(PDF)

$(PDF): $(TEX) refs.bib
	pdflatex $(TEX)
	bibtex dissertation
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f dissertation.aux dissertation.bbl dissertation.blg dissertation.log dissertation.out dissertation.toc $(PDF)

.PHONY: all clean
