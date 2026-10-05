PY ?= python3

.PHONY: install install-dev demo test all train evaluate eda federated report clean

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

demo:
	$(PY) run_all.py --demo --quick

test:
	$(PY) -m pytest -q

all:
	$(PY) run_all.py --protocol research --update-readme

eda:
	$(PY) src/eda.py --protocol research

train:
	$(PY) src/train.py --protocol research

evaluate:
	$(PY) src/evaluate.py --protocol research

federated:
	$(PY) src/federated.py --protocol research

report:
	$(PY) src/report.py --protocol research

clean:
	rm -rf .pytest_cache data/synthetic results/demo models_demo reports/demo
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
