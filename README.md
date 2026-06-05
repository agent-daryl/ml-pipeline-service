# ML Pipeline Service

End-to-end ML pipeline demonstrating MLOps best practices: data ingestion, model training, model serving with FastAPI, Prometheus metrics, and data quality monitoring.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Source │ ──▶ │  Ingestion  │ ──▶ │   Training   │ ──▶ │   Storage    │
│  (Sklearn    │     │  & Preproc  │     │  (sklearn)   │     │  (joblib)    │
│   datasets)  │     │             │     │              │     │              │
└─────────────┘     └─────────────┘     └──────────────┘     └──────────────┘
                                                                  │
┌─────────────┐     ┌─────────────┐     ┌──────────────┐         │
│ Prometheus  │ ◀── │  FastAPI    │ ◀───│   Model      │ ◀───────┘
│   Grafana   │     │  /predict   │     │   Load       │
│  (monitor)  │     │  /health    │     │              │
└─────────────┘     │  /metrics   │     └──────────────┘
                    │  /validate  │     ┌──────────────┐
                    └─────────────┘     │ Data Quality │
                                        │  Validator   │
                                        └──────────────┘
```

## Quick Start

### Local Development

```bash
python3 scripts/train.py
python3 scripts/serve.py
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 25, "AveRooms": 5, "AveBedrms": 1, "Population": 1200, "AveOccup": 3, "Latitude": 37.5, "Longitude": -122.0}'
```

### Docker

```bash
docker build -t ml-pipeline-service .
docker run -p 8000:8000 ml-pipeline-service
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict housing price from features |
| `/health` | GET | Health check with model status |
| `/metrics` | GET | Prometheus metrics |
| `/validate` | POST | Validate input data quality |

## Monitoring

- **Prometheus** metrics: request count, latency histograms, prediction distribution, data drift scores
- **Data quality**: schema validation, missing value detection, value range checks, feature drift detection

## Project Structure

```
mlops_portfolio/
├── src/
│   ├── ingestion/       # Data loading and preprocessing
│   ├── training/        # Model training pipeline
│   ├── serving/         # FastAPI application
│   └── monitoring/      # Data quality and drift detection
├── scripts/             # Entry points
├── models/              # Trained model artifacts
├── data/                # Cached datasets and stats
├── tests/               # Unit tests
├── docs/                # Architecture and runbooks
└── Dockerfile
```

## Author

Built by agent-daryl as part of Daryl Allen's MLOps portfolio for career transition from VMware/NSX to OpenShift AI.
