#!/usr/bin/env bash

f=$1
t=$2
echo "$1 \"$2\" $3"

# exit

jupyter-nbconvert-3.10  --to latex $f".ipynb"
../Utilities/filter_latex.py -i $f -t "$t"  -p -a >& tmp.tex
pdflatex "tmp.tex"; mv tmp.pdf "$f".pdf
rm tmp.[^a]*
rm -rf $f"_files" $f".tex"