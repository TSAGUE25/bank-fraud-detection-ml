# -*- coding: utf-8 -*-
"""Modèles de détection de fraude : Isolation Forest, SMOTE+XGBoost, évaluation PR-AUC."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, classification_report,
                              confusion_matrix, precision_recall_curve,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_isolation_forest(X_train, X_test, y_test,
                          contamination: float = 0.0017) -> dict:
    """Détection non-supervisée par Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X_train)
    # -1 = anomalie → 1 (fraude), 1 = normal → 0
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)
    scores  = -model.score_samples(X_test)  # score d'anomalie (plus haut = plus suspect)

    pr_auc = average_precision_score(y_test, scores)
    print(f"\n[Isolation Forest] PR-AUC = {pr_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Légitime", "Fraude"]))
    return {"model": model, "y_pred": y_pred, "scores": scores, "pr_auc": pr_auc}


def build_xgb_pipeline() -> Pipeline:
    """Pipeline StandardScaler + XGBoostClassifier avec scale_pos_weight."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=580,  # ~(n_legit / n_fraud) pour compenser le déséquilibre
            eval_metric="aucpr", random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def apply_smote(X_train, y_train, sampling_strategy: float = 0.01):
    """Suréchantillonne la classe minoritaire avec SMOTE."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=-1)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE : {y_train.sum()} → {y_res.sum()} fraudes | ratio {y_res.mean():.3%}")
        return X_res, y_res
    except ImportError:
        print("imbalanced-learn non installé — SMOTE ignoré, entraînement sur données brutes")
        return X_train, y_train


def evaluate_fraud_model(pipeline, X_test, y_test,
                          cost_fp: float = 15, cost_fn: float = 450) -> dict:
    """Évalue le modèle : PR-AUC, coût métier, seuil optimal."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Seuil optimal par rapport coût
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    costs = cost_fp * (1 - precision) + cost_fn * (1 - recall)
    best_idx   = np.argmin(costs[:-1])
    best_thr   = thresholds[best_idx]
    y_pred_opt = (y_proba >= best_thr).astype(int)

    print(f"\n[XGBoost+SMOTE] PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | Seuil optimal={best_thr:.3f}")
    print(classification_report(y_test, y_pred_opt, target_names=["Légitime", "Fraude"]))

    n_fp = confusion_matrix(y_test, y_pred_opt)[0, 1]
    n_fn = confusion_matrix(y_test, y_pred_opt)[1, 0]
    cout_total = n_fp * cost_fp + n_fn * cost_fn
    print(f"Coût total estimé : {cout_total:,.0f} € (FP×{cost_fp}€ + FN×{cost_fn}€)")

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "best_threshold": best_thr,
            "cout_total_eur": cout_total}
