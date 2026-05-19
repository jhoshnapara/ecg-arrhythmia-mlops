# Your Build Guide (3 Weekends)

> This is your hands-on plan. The repo is already built. You need to:
> 1. Run the pipeline and get real numbers
> 2. Tune anything that comes back weak
> 3. Deploy

---

## Weekend 1 — Get It Running End to End

**Goal:** working training pipeline + first set of real numbers.

### Step 1: Local setup (30 min)

```bash
git init
git add .
git commit -m "Initial commit: scaffold"

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Get the data (15 min)

```bash
python scripts/download_data.py
```

This downloads ~75 MB from PhysioNet. Sometimes their server is slow — re-run if it stalls (it's idempotent).

### Step 3: Preprocess (10 min)

```bash
python scripts/preprocess.py
```

You'll see per-record beat counts, then final class distributions. Take a screenshot of this output — it's good README material.

### Step 4: Sanity check (5 min)

```bash
pytest tests/
python src/models/cnn.py   # Should print model param count
```

### Step 5: First training run (30-60 min on CPU, 5-10 min on GPU)

```bash
# Terminal 1
mlflow ui --port 5000

# Terminal 2
python scripts/train.py --config configs/cnn_baseline.yaml
```

Open `http://localhost:5000` and watch the metrics climb.

### Step 6: Look at the results

In MLflow UI, click your run. Check:
- **`test_macro_f1`** — anywhere from 0.65 to 0.85 is normal for a first run with this dataset
- **`test_acc`** — should be 0.85-0.95 (high because Normal class dominates)
- The classification report in artifacts will show per-class precision/recall

**If macro F1 < 0.65:** the model is collapsing to Normal-only. Try:
- Increase class weights' effect (multiply weights by 2x in `train.py`)
- Lower learning rate to 0.0005
- Train longer (50 epochs)

**Write your real numbers into the README.** Replace every `_Fill in after training_` with what you actually got.

---

## Weekend 2 — Improve & Serve

**Goal:** model that performs well + working API.

### Step 1: Iterate on the model (2-4 hrs)

Try at least 2 more experiments. Suggested:
1. **Heavier regularization:** dropout=0.5
2. **Longer training:** epochs=50, lower lr=0.0005
3. **Larger model:** edit `cnn.py`, double the channels (32→64, 64→128, 128→256)

Each gets logged to MLflow automatically. Compare in the UI.

### Step 2: Promote the best model

Get the run ID of your best model from MLflow UI, then:

```bash
python scripts/promote_model.py --run-id <id> --stage Production
```

### Step 3: Build the Docker image

```bash
docker build -t ecg-api -f docker/Dockerfile .
docker run -p 8000:8000 ecg-api
```

### Step 4: Test the API

Open `http://localhost:8000/docs` — FastAPI auto-generates an interactive API explorer. Use it to test the `/predict` endpoint.

Or with curl:
```bash
# Generate a sample payload from a real test beat
python -c "
import numpy as np, json
d = np.load('data/processed/test.npz')
print(json.dumps({'signal': d['X'][0].tolist()}))
" > sample_payload.json

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_payload.json | python -m json.tool
```

### Step 5: Take a screenshot of API response

Add `docs/api_screenshot.png` showing the JSON response. This goes in your README.

---

## Weekend 3 — Deploy & Polish

**Goal:** live demo + drift monitoring + interview-ready presentation.

### Step 1: Set up Hugging Face Spaces (1 hr)

1. Sign up at `huggingface.co` (free)
2. Create a new Space → Streamlit
3. Upload these files to the Space:
   - `scripts/demo_app.py` → rename to `app.py`
   - `src/models/cnn.py` → in `src/models/`
   - Your trained weights → save as `models/best.pt`
   - `data/processed/test.npz` → in `data/processed/` (this is ~5MB, fits)
   - A minimal `requirements.txt` (only Streamlit, torch, numpy, plotly)

Push it. In 5 minutes, your Space is live at `huggingface.co/spaces/<your-username>/ecg-arrhythmia-cnn`.

**Add the URL to the top of your README.** This is the single most important thing in your portfolio.

### Step 2: Save your trained weights

Add this to the end of `train.py` (after `mlflow.pytorch.log_model`):

```python
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/best.pt")
```

Re-run training once so you have a `models/best.pt` file (also goes in `.gitignore`, so use Git LFS or HuggingFace Spaces only).

### Step 3: Run a drift report

```bash
# Simulate "production data" by taking the test set as current
python src/monitoring/drift_check.py \
    --reference data/processed/train.npz \
    --current data/processed/test.npz \
    --output reports/drift_report.html
```

Open the HTML in a browser. Take a screenshot. Add it to `docs/`.

### Step 4: Make the architecture diagram

1. Go to draw.io (free)
2. Recreate the ASCII diagram from the README as a proper diagram
3. Export as PNG → `docs/architecture.png`

### Step 5: Final README polish

Open README.md and fill in every `_Fill in after training_` with your actual numbers. Add:
- Link to your HF Spaces demo at the top
- Screenshots in a `docs/` folder
- Your name and contact info at the bottom

### Step 6: Push to GitHub

```bash
git add .
git commit -m "Complete ECG arrhythmia MLOps pipeline with deployment"
git remote add origin https://github.com/<you>/ecg-arrhythmia-mlops.git
git push -u origin main
```

**Pin this repo** on your GitHub profile (Settings → Pinned repositories).

---

## Interview Prep: What to Say

When asked "tell me about a project," use this structure:

> "I built an end-to-end MLOps pipeline for cardiac arrhythmia detection from ECG signals — the same general problem space I work in at Abbott. It uses a 1D CNN trained on the MIT-BIH database with class-weighted loss to handle the heavy imbalance toward normal beats. The interesting part wasn't just the modeling — it was the production engineering: I wired up MLflow for experiment tracking and a model registry, served the model behind a FastAPI endpoint in Docker, and added an Evidently-based drift monitor for production data. I deployed a Streamlit demo on Hugging Face Spaces so anyone can try it. Happy to dig into any layer of the stack — modeling choices, the patient-level split issue, deployment tradeoffs..."

The phrase **"happy to dig into any layer"** is intentional. It signals you actually built it and aren't afraid of any question.

Read `docs/design-decisions.md` 3-4 times. Those are the answers to the follow-up questions.

---

## What to Add to Your Resume

Replace your old "Projects" section (or add one) with:

```
ECG Arrhythmia Detection — Production MLOps Pipeline             github.com/<you>/ecg-arrhythmia-mlops
PyTorch · MLflow · FastAPI · Docker · Hugging Face Spaces
• Built 1D CNN classifier on MIT-BIH database achieving <YOUR_F1> macro-F1 on
  the standard DS1/DS2 patient-level split
• Implemented full MLOps stack: MLflow experiment tracking + model registry,
  FastAPI inference service in Docker, Evidently-based drift monitoring
• Deployed live interactive demo on Hugging Face Spaces with Streamlit
  frontend; <YOUR_LATENCY>ms CPU inference latency
```

Customize with your actual numbers after training.

---

## Common Pitfalls

1. **Don't skip the patient-level split.** Random splits give you 99% accuracy and lying numbers. Every senior interviewer will catch this.
2. **Don't claim numbers you didn't measure.** If your test F1 is 0.71, write 0.71. "Reduced false positives by 70%" is your Abbott bullet — your project bullet is real measured F1.
3. **Don't forget the live demo link.** A README without a deployed demo is half the project.
4. **Don't push raw data or `mlruns/` to GitHub.** They're in `.gitignore` for a reason — they make the repo huge and slow to clone.
