"""Tune two Random Forest pipelines and track every candidate with MLflow."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tourism_project.schema import (
    CATEGORY_OPTIONS,
    DATA_PATH,
    POST_PITCH_FEATURES,
    PRE_CONTACT_FEATURES,
    ROOT,
    TARGET,
)

PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, None],
    "classifier__min_samples_leaf": [1, 4],
    "classifier__class_weight": [None, "balanced"],
}
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def make_model(feature_names):
    categorical = [name for name in feature_names if name in CATEGORY_OPTIONS]
    numeric = [name for name in feature_names if name not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=1)),
        ]
    )


def evaluate(target, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            target,
            predicted,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }


def load_splits(splits_dir):
    summary = json.loads((splits_dir / "preparation_summary.json").read_text())
    if summary["source_sha256"] != hashlib.sha256(DATA_PATH.read_bytes()).hexdigest():
        raise ValueError("The split artifact was prepared from a different CSV.")
    for name, expected_hash in summary["files_sha256"].items():
        if (
            hashlib.sha256((splits_dir / name).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise ValueError(f"Split artifact fingerprint mismatch: {name}")
    loaded = {
        name: pd.read_csv(splits_dir / f"{name}.csv")
        for name in [
            "Xtrain",
            "Xtest",
            "ytrain",
            "ytest",
            "groups_train",
            "groups_test",
        ]
    }
    for split in ["train", "test"]:
        sizes = [
            len(loaded[f"X{split}"]),
            len(loaded[f"y{split}"]),
            len(loaded[f"groups_{split}"]),
        ]
        if len(set(sizes)) != 1:
            raise ValueError(f"Mismatched feature, target, or group rows in {split}.")
    if set(loaded["groups_train"].iloc[:, 0]) & set(loaded["groups_test"].iloc[:, 0]):
        raise ValueError("Profile leakage detected in the split artifact.")
    return loaded, summary


def train_models(splits_dir=None, reports_dir=None):
    splits_dir = Path(splits_dir or ROOT / "artifacts" / "data-splits")
    reports_dir = Path(reports_dir or ROOT / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir = ROOT / "tourism_project" / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    loaded, preparation = load_splits(splits_dir)
    ytrain = loaded["ytrain"][TARGET]
    ytest = loaded["ytest"][TARGET]
    groups = loaded["groups_train"].iloc[:, 0]

    mlflow.set_tracking_uri(f"sqlite:///{(reports_dir / 'mlflow.db').resolve()}")
    experiment_name = "wellness-tourism-purchases"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=str((reports_dir / "mlflow-artifacts").resolve()),
        )
    mlflow.set_experiment(experiment_name)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(loaded["Xtrain"], ytrain)
    results = {
        "selection_metric": "Grouped cross-validation average precision",
        "threshold": 0.5,
        "baseline": evaluate(ytest, baseline.predict_proba(loaded["Xtest"])[:, 1]),
        "test_rows": len(ytest),
        "test_purchase_rate": float(ytest.mean()),
        "source_sha256": preparation["source_sha256"],
        "models": {},
    }
    predictions = {}
    metadata = None
    for scope, feature_names in [
        ("pre_contact", PRE_CONTACT_FEATURES),
        ("post_pitch", POST_PITCH_FEATURES),
    ]:
        Xtrain = loaded["Xtrain"][feature_names]
        Xtest = loaded["Xtest"][feature_names]
        search = GridSearchCV(
            make_model(feature_names),
            PARAM_GRID,
            scoring=SCORING,
            refit="average_precision",
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=2,
            error_score="raise",
        )
        with mlflow.start_run(run_name=scope) as parent:
            mlflow.log_params(
                {
                    "scope": scope,
                    "random_state": 42,
                    "cv_folds": 3,
                    "selection_metric": "average_precision",
                    "threshold": 0.5,
                    "feature_count": len(feature_names),
                    "source_sha256": preparation["source_sha256"],
                }
            )
            mlflow.log_dict(PARAM_GRID, "parameter_grid.json")
            mlflow.log_dict({"feature_names": feature_names}, "features.json")
            search.fit(Xtrain, ytrain, groups=groups)
            trial_rows = []
            for index, parameters in enumerate(search.cv_results_["params"]):
                row = {
                    "candidate": index + 1,
                    **parameters,
                    "rank": int(
                        search.cv_results_["rank_test_average_precision"][index]
                    ),
                }
                with mlflow.start_run(
                    run_name=f"{scope}_candidate_{index + 1:02d}", nested=True
                ) as trial:
                    mlflow.log_params(parameters)
                    for metric in SCORING:
                        mean = float(search.cv_results_[f"mean_test_{metric}"][index])
                        std = float(search.cv_results_[f"std_test_{metric}"][index])
                        mlflow.log_metric(f"cv_{metric}", mean)
                        mlflow.log_metric(f"cv_{metric}_std", std)
                        row[f"cv_{metric}"] = mean
                        row[f"cv_{metric}_std"] = std
                    row["mlflow_run_id"] = trial.info.run_id
                trial_rows.append(row)
            trials_path = reports_dir / f"cv_{scope}.csv"
            pd.DataFrame(trial_rows).sort_values("rank").to_csv(
                trials_path, index=False
            )
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("best_cv_average_precision", search.best_score_)
            best = search.best_estimator_
            probabilities = best.predict_proba(Xtest)[:, 1]
            predictions[scope] = probabilities
            metrics = evaluate(ytest, probabilities)
            for metric in SCORING:
                mlflow.log_metric(f"test_{metric}", metrics[metric])
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            model_path = (
                deployment_dir / "model.joblib"
                if scope == "pre_contact"
                else reports_dir / "post_pitch_model.joblib"
            )
            joblib.dump(best, model_path, compress=3)
            mlflow.log_artifact(str(model_path), artifact_path="saved_model")
            mlflow.log_artifact(str(trials_path))
            mlflow.log_dict(metrics, "test_evaluation.json")
            results["models"][scope] = {
                "feature_count": len(feature_names),
                "best_parameters": search.best_params_,
                "best_cv_average_precision": float(search.best_score_),
                "candidate_count": len(trial_rows),
                "mlflow_parent_run_id": parent.info.run_id,
                "test_metrics": metrics,
            }
            if scope == "pre_contact":
                metadata = {
                    "model_scope": scope,
                    "feature_names": feature_names,
                    "threshold": 0.5,
                    "source_sha256": preparation["source_sha256"],
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "packages": {
                        name: version(name)
                        for name in ["scikit-learn", "numpy", "pandas", "joblib"]
                    },
                    "best_parameters": search.best_params_,
                    "best_cv_average_precision": float(search.best_score_),
                    "test_metrics": metrics,
                    "numeric_training_ranges": {
                        name: [float(Xtrain[name].min()), float(Xtrain[name].max())]
                        for name in feature_names
                        if name not in CATEGORY_OPTIONS
                    },
                }
            print(f"\n{scope}: best parameters = {search.best_params_}")
            print(
                classification_report(
                    ytest,
                    probabilities >= 0.5,
                    labels=[0, 1],
                    zero_division=0,
                )
            )
    (deployment_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (reports_dir / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for scope, probabilities in predictions.items():
        PrecisionRecallDisplay.from_predictions(
            ytest, probabilities, name=scope, ax=axes[0]
        )
        RocCurveDisplay.from_predictions(ytest, probabilities, name=scope, ax=axes[1])
    axes[0].axhline(ytest.mean(), color="gray", linestyle="--", label="Prevalence")
    axes[0].legend()
    axes[0].set_title("Held-out precision-recall comparison")
    axes[1].set_title("Held-out ROC comparison")
    fig.tight_layout()
    fig.savefig(reports_dir / "performance_curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (scope, probabilities) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            ytest,
            probabilities >= 0.5,
            display_labels=["No purchase", "Purchase"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(scope)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrices.png", dpi=140)
    plt.close(fig)
    print("\nEvaluation written to reports/evaluation.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir", type=Path, default=ROOT / "artifacts" / "data-splits"
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    train_models(args.splits_dir, args.reports_dir)
