# DentalScan AI - reproduction entry points.
# Every target is safe to re-run; nothing here trains unless you ask it to.
#
# Four CLIs do the work, each with subcommands and --help:
#   dataset.py       prepare | audit | folds
#   model.py         train | ablate | aggregate
#   evaluate.py      per-class | compare
#   evaluate_new.py  errors | sweep | explain

PY      ?= python
DATASET ?= $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('configs/data.yaml'))['path'])" 2>/dev/null)
DEVICE  ?= 0
SEEDS   ?= 0 1 2

.PHONY: help setup data audit folds test compare train ablation-dry ablation report all clean-results

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:          ## install the research dependencies
	pip install -r requirements-research.txt

data:           ## locate the dataset and write configs/data.yaml
	$(PY) dataset.py prepare --search-root . --also-single-copy

audit: data     ## per-class support, leakage check, imbalance
	$(PY) dataset.py audit --dataset-root "$(DATASET)"

folds: data     ## build grouped, rarity-aware 5-fold CV splits
	$(PY) dataset.py folds --folds 5 \
	  --image-dirs "$(DATASET)/train/images" "$(DATASET)/valid/images" "$(DATASET)/test/images"

test:           ## run the metric test suite
	$(PY) -m pytest tests/ -q

compare:        ## regenerate figures and tables from runs already on disk
	$(PY) evaluate.py compare \
	  --runs runs/detect/Yolo_8n_250 runs/detect/Yolo_10s_train runs/detect/Yolo_12m_250epochs \
	  --labels YOLOv8n YOLOv10s YOLOv12m

train:          ## train the baseline across $(SEEDS)  [GPU, hours]
	$(PY) model.py train --name baseline --model yolov10s.pt --seeds $(SEEDS) --device $(DEVICE)

ablation-dry:   ## show what the ablation grid would run, and what it costs
	$(PY) model.py ablate --dry-run

ablation:       ## run the ablation grid  [GPU, tens of hours]
	$(PY) model.py ablate --seeds $(SEEDS) --device $(DEVICE)
	$(PY) model.py aggregate --manifest results/ablation_manifest.json

report:         ## rebuild the technical report PDF
	cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	  bibtex main >/dev/null && \
	  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode main.tex >/dev/null && \
	  echo "report/main.pdf"

all: audit folds test compare  ## everything that does not need a GPU

clean-results:  ## remove generated results (keeps runs/ and weights)
	rm -rf results/figures results/*.json results/aggregate results/error_analysis
