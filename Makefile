# All logic lives in scripts/tasks.py (works on Windows via make.cmd too).
PY ?= python3
TASKS := setup models frontend serve demo dev test lint typecheck bench eval-custom eval-synthetic report docker-up

.PHONY: $(TASKS) help
help:
	@$(PY) scripts/tasks.py

$(TASKS):
	$(PY) scripts/tasks.py $@
