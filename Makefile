.PHONY: venv install train-v01 train-v02 api test lint format docker

venv:
	python3.12 -m venv .venv

install:
	pip install -r requirements.txt

train-v01:
	python training/train_v01.py

train-v02:
	python training/train_v02.py

api:
	uvicorn src.app:app --port 8080

test:
	pytest -q

lint:
	ruff check .
	black --check .

format:
	black .

docker:
	docker build -t ghcr.io/<org>/<repo>:v0.1 .
