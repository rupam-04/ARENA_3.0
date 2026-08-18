.ONESHELL:
SHELL := /bin/bash

.PHONY: start

start:
	set -e
	echo "Creating conda environment ARENA with Python 3.11"
	source "$$(conda info --base)/etc/profile.d/conda.sh"
	conda create -n ARENA python=3.11 -y
	conda activate ARENA
	conda install jupyter ipykernel -y
	python -m ipykernel install --user --name ARENA --display-name "Python 3.11 (ARENA)"
	uv pip install -r requirements.txt
	cd chapter1_transformer_interp/exercises
	git clone https://github.com/saprmarks/geometry-of-truth.git
	git clone https://github.com/ApolloResearch/deception-detection.git
	cd ../..
	python -m pip install -U ipywidgets
	python -m pip install -U jupyter jupyterlab notebook