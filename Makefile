PY ?= /home/tsmc193/GraphCAD/miniconda3/envs/spike/bin/python

.PHONY: smoke nightly test report

smoke:
	PYTHONPATH=. $(PY) -m asfuzz.cli run --config configs/smoke.yaml

nightly:
	PYTHONPATH=. $(PY) -m asfuzz.cli run --config configs/nightly.yaml

test:
	PYTHONPATH=. $(PY) -m pytest -q

report:
	PYTHONPATH=. $(PY) -m asfuzz.cli report --summary runs/latest/summary.json --out runs/latest/report.html

