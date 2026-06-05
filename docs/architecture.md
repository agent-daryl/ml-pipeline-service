# ML Pipeline Service — Architecture

## Design Decisions

### Why California Housing Dataset?
- Built into scikit-learn, no external downloads needed
- 8 numerical features, well-documented
- Regression task maps cleanly to real-world inference patterns
- 20,640 samples provide meaningful train/test split

### Why Gradient Boosting?
- Strong baseline performance (R2 ~0.83 on test)
- Training time under 30 seconds on modest hardware
- No hyperparameter tuning needed for portfolio demo
- Feature importance available for explainability

### Why FastAPI?
- Native async support, automatic OpenAPI docs at /docs
- Pydantic validation built-in
- Lightweight, container-friendly
- Industry standard for ML serving

### Why Prometheus?
- Standard observability in K8s/OpenShift ecosystems
- Native Histogram and Counter types for ML metrics
- Grafana dashboards available out of the box
- Relevant to Daryl's OpenShift AI career pivot

## Pipeline Flow

```
1. Data Ingestion (src/ingestion/)
   - Fetch California Housing via sklearn.datasets
   - Drop missing values, shuffle with fixed seed
   - 80/20 train/test split
   - Compute and persist baseline statistics

2. Training (src/training/)
   - StandardScaler for feature normalization
   - GradientBoostingRegressor (200 estimators, depth 5)
   - Evaluate on holdout test set
   - Persist model + scaler as joblib artifact
   - Persist metrics + stats as JSON metadata

3. Serving (src/serving/)
   - FastAPI app with 4 endpoints:
     /health    — GET, model status and uptime
     /predict   — POST, prediction with latency tracking
     /validate  — POST, data quality and drift check
     /metrics   — GET, Prometheus exposition format
   - Pydantic schemas enforce input constraints
   - Prometheus counters, histograms for observability

4. Monitoring (src/monitoring/)
   - DataValidator compares inputs against baseline stats
   - Z-score based outlier detection per feature
   - Range checks against training distribution
   - Aggregate drift score for batch validation
```

## Deployment Targets

### Development (Current)
```bash
python3 scripts/train.py
python3 scripts/serve.py
```

### Docker
```bash
docker build -t ml-pipeline-service .
docker run -p 8000:8000 ml-pipeline-service
```

### OpenShift (Future)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline-service
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: ml-service
        image: ml-pipeline-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests: {cpu: 250m, memory: 512Mi}
          limits:   {cpu: 1000m, memory: 1Gi}
        readinessProbe:
          httpGet: {path: /health, port: 8000}
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-pipeline-service
spec:
  selector: {app: ml-pipeline-service}
  ports:
  - port: 80
    targetPort: 8000
```

## Prometheus Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `ml_pipeline_requests_total` | Counter | endpoint, status | Total requests by endpoint and outcome |
| `ml_pipeline_prediction_latency_seconds` | Histogram | — | P50/P95/P99 prediction latency |
| `ml_pipeline_prediction_value` | Histogram | — | Distribution of predicted house values |
| `ml_pipeline_data_drift_score` | Histogram | — | Drift score distribution from /validate |

## Future Enhancements
- [ ] Model versioning with MLflow
- [ ] A/B testing framework for model canary deployments
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Automated retraining on schedule
- [ ] Grafana dashboard JSON export
- [ ] OpenShift ServiceMesh integration
- [ ] Feature store integration (Feast)
