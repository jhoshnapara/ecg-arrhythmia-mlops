"""
Promote a training run's model to the MLflow Model Registry.

Why this exists: deploying directly from a run artifact is fragile.
The registry is the single source of truth — every served model is
linked to its run, its metrics, and its lifecycle stage.

Usage:
    python scripts/promote_model.py --run-id <id> --stage Staging
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ecg-arrhythmia-cnn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID to promote")
    parser.add_argument("--stage", default="Staging", choices=["Staging", "Production", "Archived"])
    args = parser.parse_args()

    client = MlflowClient()
    model_uri = f"runs:/{args.run_id}/model"

    # Register if first time, else create new version
    try:
        client.get_registered_model(MODEL_NAME)
    except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
        client.create_registered_model(MODEL_NAME)
        print(f"Created registered model: {MODEL_NAME}")

    mv = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=args.run_id)
    print(f"Registered version {mv.version}")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage=args.stage,
        archive_existing_versions=(args.stage == "Production"),
    )
    print(f"✅ Model {MODEL_NAME} v{mv.version} → {args.stage}")


if __name__ == "__main__":
    main()
