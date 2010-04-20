

all:yael.pdf reference

yael.pdf: yael.tex
	pdflatex yael

reference: 
	doxygen Doxyfile

clean: 
	rm -fr html latex *.aux *.toc *.log *~

