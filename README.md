# ECG Arrhythmia Detection — Production MLOps Pipeline

> **End-to-end MLOps pipeline for detecting cardiac arrhythmias from ECG signals using a 1D CNN, with experiment tracking, model versioning, REST API serving, and drift monitoring.**

**🔗 Live Demo:** [Add your Hugging Face Spaces URL here after deploying]
**📊 Architecture Diagram:** [See `docs/architecture.png`]

---

## Why this project

Cardiac arrhythmias are detected today using rule-based algorithms on Implantable Cardiac Monitors and Holter devices. These produce ~70% false positive rates, leading to alert fatigue in clinical workflows. This project demonstrates a deep learning approach that maintains high sensitivity while reducing false positives — and, more importantly, shows the **production engineering** required to ship such a model: experiment tracking, model registry, containerized serving, and drift monitoring.

## Results

| Metric | Value |
|--------|-------|
| Overall accuracy | _Fill in after training_ |
| Macro F1-score | _Fill in after training_ |
| Precision (V — Ventricular) | _Fill in after training_ |
| Recall (V — Ventricular) | _Fill in after training_ |
| Inference latency (CPU, single beat) | _Fill in after benchmarking_ |
| Model size | _Fill in after training_ |

_Trained on the MIT-BIH Arrhythmia Database (PhysioNet). Patient-level train/test split._

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PhysioNet   │───▶│  Data prep   │───▶│   PyTorch    │───▶│   MLflow     │
│   MIT-BIH    │    │  & features  │    │  CNN training│    │  tracking    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Hugging    │◀───│   Docker     │◀───│   FastAPI    │◀───│   MLflow     │
│  Face Spaces │    │  container   │    │  inference   │    │  model       │
│  (Streamlit) │    │              │    │   service    │    │  registry    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  Monitoring  │
                                        │ (Evidently)  │
                                        └──────────────┘
```

## Tech Stack

- **Modeling:** PyTorch, NumPy
- **Data:** WFDB (PhysioNet's tooling), Pandas, scikit-learn
- **MLOps:** MLflow (experiment tracking + model registry)
- **Serving:** FastAPI, Uvicorn, Pydantic
- **Packaging:** Docker
- **Monitoring:** Evidently AI (data drift)
- **Deployment:** Hugging Face Spaces (Streamlit frontend)

## Key Design Decisions

These are the questions an interviewer will probe. Each is documented in `docs/design-decisions.md`:

1. **Patient-level split, not beat-level.** A naive random split puts beats from the same patient in both train and test, inflating accuracy by ~10-15%. We split by patient ID so the model is evaluated on never-seen subjects — closer to clinical deployment.
2. **1D CNN over LSTM.** For single-beat classification (~360 samples per beat), CNNs are faster to train, lower latency at inference, and competitive with LSTMs on this dataset. LSTM would matter more for sequence-level (multi-beat) classification.
3. **Class weighting + focal loss.** The MIT-BIH dataset is heavily imbalanced (~90% Normal beats). We use weighted cross-entropy with class frequencies, and optionally focal loss for the minority classes (V, S).
4. **Optimize for recall on V/S classes.** Missing a real ventricular ectopic beat is far worse than a false alarm. We tune the decision threshold per-class to maximize recall at acceptable precision.
5. **MLflow registry as source of truth.** No model is deployed without being registered, tagged with a stage (`Staging`, `Production`), and linked to the exact training run.

## Project Structure

```
ecg-arrhythmia-mlops/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # PyTorch model definitions
│   ├── training/       # Training loop + MLflow integration
│   ├── inference/      # FastAPI service
│   └── monitoring/     # Drift detection
├── configs/            # Hydra-style YAML configs
├── scripts/            # Data download, training entrypoints
├── tests/              # pytest tests
├── docker/             # Dockerfile for serving
├── docs/               # Architecture, design decisions
└── notebooks/          # EDA and experiment notebooks
```

## Quick Start

### 1. Setup

```bash
git clone <this-repo>
cd ecg-arrhythmia-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data

```bash
python scripts/download_data.py
```

This downloads the MIT-BIH Arrhythmia Database (~75 MB) from PhysioNet.

### 3. Preprocess

```bash
python scripts/preprocess.py
```

Segments ECG records into individual beats, applies bandpass filtering, and creates train/val/test splits at the patient level. Output: `data/processed/{train,val,test}.npz`.

### 4. Train

```bash
mlflow ui --port 5000 &   # In one terminal
python scripts/train.py --config configs/cnn_baseline.yaml
```

Open `http://localhost:5000` to watch experiments live.

### 5. Serve

```bash
python scripts/promote_model.py --run-id <best-run-id>  # Register to MLflow model registry
docker build -t ecg-api -f docker/Dockerfile .
docker run -p 8000:8000 ecg-api
```

Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

### 6. Run the Streamlit demo

```bash
streamlit run scripts/demo_app.py
```

## What's NOT in this project (and why)

Production-ready ML systems are large. To keep this project focused, the following are documented as "future work" rather than implemented:

- Kubernetes orchestration (Docker is enough to demonstrate the pattern)
- Multi-region failover
- A/B testing infrastructure
- FHIR / HL7 integration (would be needed for real EHR deployment)

The point of this repo is to demonstrate end-to-end ownership of an ML system, not to ship a hospital-grade product.

## License

MIT. Data is from PhysioNet, governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mitdb/).
