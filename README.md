# ECG Arrhythmia Detection

5-class heartbeat classifier (Normal, Supraventricular, Ventricular, Fusion, Unclassified) trained on MIT-BIH. Includes the full serving pipeline: MLflow tracking, FastAPI inference, Docker.

## Results

![Test results](docs/run2_tuned.png)

DS1/DS2 patient-level split (test patients never seen in training).

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| N | 0.95 | 0.92 | 0.93 | 44,235 |
| S | 0.03 | 0.03 | 0.03 | 1,837 |
| V | 0.60 | 0.83 | 0.70 | 3,220 |
| F | 0.00 | 0.00 | 0.00 | 388 |
| Q | 0.00 | 0.00 | 0.00 | 7 |

Overall accuracy 0.87. V-class recall is what matters clinically (false negatives carry higher cost). S and F are weak — single-lead single-beat input with no RR features. Threshold tuning per class is a deployment-time decision, not retrained for argmax.

## Two runs

Ran inverse-frequency class weights first. Confusion matrix showed it was over-predicting V everywhere (N precision tanked to 0.95 with recall only 0.80 - lots of N being flagged as something else). Switched to sqrt-scaled weights, retrained. N-class F1 went 0.87 -> 0.93, V-recall held at 0.83. Both runs in MLflow.

## API

![health](docs/api_swagger_health.png)
![model-info](docs/api_swagger_model_info.png)

FastAPI. Endpoints: `/health`, `/model-info`, `/predict`. Inference 16ms CPU, +13ms in Docker.

## Stack

PyTorch, MLflow, FastAPI, Docker, Evidently, Streamlit. Model is 44K params, CPU-only.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/preprocess.py

mlflow ui --port 5000 &
python scripts/train.py --config configs/cnn_baseline.yaml

uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

Docker:
```bash
docker build -t ecg-api -f docker/Dockerfile .
docker run -p 8000:8000 ecg-api
```

## Data

[MIT-BIH](https://physionet.org/content/mitdb/). 48 records, 360 Hz, beat-level annotations.

## Notes

Not for clinical use. Demo / portfolio project.
