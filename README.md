# AI381 Lab 2: Container + CI

[![CI](https://github.com/RoaaFadel/mlops-lab2./actions/workflows/ci.yml/badge.svg)](https://github.com/RoaaFadel/mlops-lab2./actions/workflows/ci.yml)

A small Iris classifier served with FastAPI, packaged in a multi-stage, non-root Docker image, and built, scanned (Trivy) and published to GHCR by GitHub Actions.

## Run it with one command

```bash
docker build -t iris-api . && docker run --rm -p 8000:8000 iris-api
```

Then, in another terminal:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service and model status |
| POST | `/predict` | Predict the iris class from 4 measurements (cm) |

## Development

```bash
pip install -r requirements-dev.txt
python -m src.train.train
python -m pytest -v
ruff check .
```

## CI

- Pull requests: lint (ruff) and tests (pytest).
- Push to `main`: build the image, scan it with Trivy, push it to GHCR tagged with the commit SHA.
- Third-party actions are pinned to full commit SHAs, and the base images are pinned by digest.
