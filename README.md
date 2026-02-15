# Cats vs Dogs – End-to-End MLOps Pipeline

##  Project Overview

This project implements a complete MLOps lifecycle for a Cats vs Dogs image classification model.

It covers:

- Model training
- Model packaging
- Containerization
- Continuous Integration (CI)
- Continuous Deployment (CD)
- Post-deployment monitoring
- Model performance tracking

The system automatically builds, tests, pushes, deploys, and monitors a Dockerized inference service.

---

#  Architecture Overview

    ┌──────────────┐
    │   Developer  │
    └──────┬───────┘
           │ Push to main
           ▼
    ┌──────────────────┐
    │ GitHub Actions CI│
    ├──────────────────┤
    │ Run Tests        │
    │ Build Docker     │
    │ Push to GHCR     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ GitHub Container │
    │ Registry (GHCR)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Self-Hosted      │
    │ Runner (Windows) │
    ├──────────────────┤
    │ docker compose   │
    │ Pull latest      │
    │ Deploy container │
    │ Smoke tests      │
    └──────────────────┘


#  Module Breakdown

## M1 – Model Training

- CNN-based PyTorch model
- Saved as `model.pt`
- Stored under `artifacts/`
- Deterministic split using fixed seed

## M2 – Packaging & Containerization

### Inference API
- Framework: **FastAPI**
- Endpoints:
  - `GET /health`
  - `POST /predict`
  - `POST /feedback`
  - `GET /metrics`

### Environment
- Fully pinned `requirements.txt`
- Reproducible dependency versions

### Docker
- Image built using `Dockerfile`
- Uvicorn exposed on `0.0.0.0:8000`
- Production-ready container

## M3 – Continuous Integration (CI)

Implemented using **GitHub Actions**.

### Pipeline Steps
- Install dependencies
- Run `pytest`
- Build Docker image
- Push image to GHCR

### Image Registry
- GitHub Container Registry (GHCR)

## M4 – Continuous Deployment (CD)

### Deployment Target
- Local Windows machine
- Self-hosted GitHub Actions runner
- Docker Compose deployment

### CD Flow
- Triggered on `main` branch push
- Pull latest image from GHCR
- Deploy using `docker compose`
- Run automated smoke tests

### Smoke Tests
- Health check (`/health`)
- Prediction test (`/predict`)
- Pipeline fails if service does not start

## M5 – Monitoring & Performance Tracking

### Basic Monitoring

Implemented via middleware:

- Request path logging
- HTTP status logging
- Latency tracking
- Request counter


### Metrics Endpoint

Returns:(Sample)
```json
{
  "total_requests": 10,
  "average_latency": 0.034,
  "total_predictions": 5,
  "correct_predictions": 4,
  "accuracy": 0.8
}

### Post-Deployment Performance Tracking
POST /feedback
{
  "predicted_label": "cat",
  "true_label": "cat"
}

### Used to:
Collect simulated real-world labels
Track model accuracy dynamically
Monitor performance degradation

### LOCAL Testing
## Start Service
docker compose up -d

#Local Health check URL
http://localhost:8000/health

This project demonstrates a complete, production-style MLOps workflow:

Model → API → Container → CI → Registry → CD → Monitoring → Feedback Loop

All stages are automated and reproducible.

