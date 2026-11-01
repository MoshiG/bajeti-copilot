.PHONY: install install-all test lint ping fetch ingest index index-smoke gold-check eval-retrieval eval demo clean

CONFIG ?= configs/default.yaml

install:        ## everything except PyTorch (enough for tests, CI and BM25)
	pip install -e ".[dev,ingest,index]"

install-all:    ## everything, for a GPU pod or a full local run
	pip install -e ".[all]"

test:
	pytest --cov=bajeti --cov-report=term-missing

lint:
	ruff check src tests

ping:           ## check the LLM endpoint answers
	bajeti -c $(CONFIG) ping

fetch:          ## download the corpus in data/corpus.yaml into data/raw/
	bajeti -c $(CONFIG) fetch
ingest:         ## parse PDFs, OCR scans, write naive + structured chunks
	bajeti -c $(CONFIG) ingest
index:          ## embed chunks (bge-m3, needs [embed]) and build the budget database
	bajeti -c $(CONFIG) index
index-smoke:    ## same pipeline with hashing vectors: no GPU, no model download
	bajeti -c $(CONFIG) index --embedder hashing
gold-check:     ## validate eval/gold.jsonl
	bajeti -c $(CONFIG) gold-check
eval-retrieval: ## retrieval metrics for all five configs: no LLM; needs .[embed], CPU is fine
	bajeti -c $(CONFIG) ablate --retrieval-only
eval:           ## full ablation on the test split (needs the LLM endpoint); resumable
	bajeti -c $(CONFIG) ablate --split test --resume
demo:           ## launch the Gradio demo on http://127.0.0.1:7860
	bajeti -c $(CONFIG) demo

clean:
	rm -rf data/processed/* data/index/* eval/results/* .pytest_cache .coverage
