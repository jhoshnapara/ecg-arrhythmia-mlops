# Design Decisions

> Every choice here is something an interviewer will probe. Read this before any technical screen.

## 1. Why patient-level split, not random?

**The naive approach:** randomly split all beats 80/20 into train/test.

**The problem:** the same patient's beats end up in both sets. The model learns "this is patient 215's heartbeat shape" and gets near-perfect test accuracy that completely vanishes on new patients. Published papers using random splits typically report 99%+ accuracy. Papers using patient-level splits report 85-93%. The latter is what's clinically relevant.

**Our approach:** we use the **DS1/DS2 split from Chazal et al. (2004)**, which holds out 22 entire patient records as the test set. This is the standard benchmark in arrhythmia detection literature, so our numbers are comparable to published work.

**Interview probe:** "What if a patient in DS2 has a similar arrhythmia profile to someone in DS1?" — Answer: that's the realistic scenario at deployment. A new patient at a new hospital will share *some* characteristics with training patients but not all. The model needs to generalize to that, not memorize per-patient features.

## 2. Why a 1D CNN and not an LSTM or Transformer?

**For single-beat classification, CNNs win on:**
- **Latency:** ~1-2ms per beat on CPU vs ~5-10ms for an LSTM of equivalent capacity
- **Compute on edge:** the model is meant to run on Implantable Cardiac Monitors with severely limited compute. CNN inference is dominated by FLOPs which are predictable; LSTM inference has data-dependent sequential dependencies that are harder to optimize.
- **Receptive field is enough:** a 1-second window at 360 Hz is short enough that a 3-layer CNN sees the whole beat.

**Where LSTM/Transformer would win:** multi-beat sequence-level classification (rhythm analysis over 10+ beats), where temporal dependencies between beats matter. That's a different problem.

**Interview probe:** "What about a Transformer?" — They're overkill for ~360-timestep sequences and add significant latency. A CNN is the right tool. State-space models (Mamba) could be a future direction for multi-beat extension.

## 3. Why class-weighted cross-entropy?

The MIT-BIH dataset is extremely imbalanced: about 90% Normal, ~7% Ventricular, ~2.5% Supraventricular, <1% Fusion. Without weighting, the model learns to predict "Normal" for everything and gets 90% accuracy with 0% recall on the clinically important V class.

We weight each class by `total / (n_classes × class_count)`. Minority classes get weights of 10-40x the Normal class, forcing the loss to penalize errors on them.

**Alternatives we considered:**
- **Oversampling minority classes:** works, but doesn't expose the model to new data, just duplicates. Easier to overfit.
- **SMOTE:** creates synthetic beats that aren't physiologically realistic — bad for ECG.
- **Focal loss:** automatically downweights easy examples. Strong choice; included as a config option.

**Interview probe:** "Why not just use focal loss?" — Class weights are simpler, more interpretable, and the dataset's imbalance is severe enough that explicit weights are the right starting point. Focal loss is a refinement, not a replacement.

## 4. Why per-class threshold tuning?

In healthcare AI, **the cost of false negatives is far higher than the cost of false positives** for life-threatening classes. Missing a real ventricular arrhythmia could mean missing a precursor to cardiac arrest. A false alarm just means the cardiologist takes another look.

We tune the decision threshold per class on the validation set to **maximize recall at 95% precision** for the V and S classes. For Normal beats, default threshold is fine.

**Interview probe:** "What if your false positive rate goes too high?" — That's the tradeoff. In practice, you'd work with clinical teams to set the operating point. The right metric isn't a single number — it's the precision-recall curve, and you pick the point on it that matches clinical needs.

## 5. Why MLflow over W&B / Neptune / Comet?

- **MLflow is open source and free to self-host.** No vendor lock-in.
- **The Model Registry is critical:** it's a database that tracks every version of every model, what training run produced it, what stage it's in. It's the source of truth for what's deployed.
- **W&B has nicer UI** but the registry story is less mature.
- **Most healthcare/regulated companies (including Abbott) use MLflow or in-house equivalents.**

## 6. Why FastAPI?

- Native async support: handles concurrent inference requests well
- Pydantic validation: catches malformed inputs before they reach the model
- Auto-generated OpenAPI docs: free contract for clients
- Same performance as Flask + Gunicorn, with way less boilerplate

## 7. Why Evidently for drift monitoring?

- Open source, integrates with any logging stack
- Built-in stat tests: Wasserstein, K-S, PSI — appropriate for different feature types
- HTML reports are reviewable by non-engineers (clinical or compliance teams)

## 8. What would I change at production scale?

- **Replace MLflow file-store with Postgres + S3:** file-store doesn't handle concurrent writes
- **Use BentoML or Triton instead of raw FastAPI:** better for multi-model serving, batching, GPU inference
- **Add a feature store (Feast):** if features became more complex (patient history, prior diagnoses)
- **Move drift monitoring to Airflow/Prefect:** scheduled runs with alerting integration
- **Add shadow deployment:** new model serves alongside current model, compare predictions before promoting

## 9. Things I deliberately did NOT do (and why)

- **Did not chase 99% accuracy:** that's the random-split number. With patient-level split, 90% is good. Chasing the wrong number is worse than reporting the right one.
- **Did not use transfer learning from larger models:** for 360-sample 1D signals, ImageNet-pretrained models don't help. There's no ECG equivalent of BERT yet (PTB-XL pretraining exists but adds complexity).
- **Did not deploy on Kubernetes:** Docker alone is enough to demonstrate the pattern. K8s adds operational complexity that doesn't serve the demo.
