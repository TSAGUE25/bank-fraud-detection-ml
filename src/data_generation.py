# -*- coding: utf-8 -*-
"""Génération de données simulées de transactions bancaires (fraude ~0.17%)."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_fraud_data(n_total: int = 500_000, n_fraud: int = 850,
                        seed: int = 42) -> pd.DataFrame:
    """Génère un dataset de transactions avec classe ultra-rare (fraude ~0.17%)."""
    np.random.seed(seed)
    legit = pd.DataFrame({
        "montant":           np.abs(np.random.lognormal(3.5, 1.5, n_total - n_fraud)),
        "heure":             np.random.randint(0, 24, n_total - n_fraud),
        "pays_code":         np.random.choice([0, 1, 2, 3], n_total - n_fraud,
                                               p=[0.80, 0.10, 0.05, 0.05]),
        "nb_trans_24h":      np.random.poisson(3, n_total - n_fraud),
        "dist_km":           np.abs(np.random.normal(50, 30, n_total - n_fraud)),
        "ancien_client_ans": np.random.randint(0, 20, n_total - n_fraud),
        "fraude":            0,
    })
    fraud = pd.DataFrame({
        "montant":           np.abs(np.random.lognormal(5.5, 2.0, n_fraud)),
        "heure":             np.random.choice([0, 1, 2, 3, 22, 23], n_fraud),
        "pays_code":         np.random.choice([1, 2, 3], n_fraud, p=[0.40, 0.35, 0.25]),
        "nb_trans_24h":      np.random.poisson(8, n_fraud),
        "dist_km":           np.abs(np.random.normal(2500, 800, n_fraud)),
        "ancien_client_ans": np.random.randint(0, 3, n_fraud),
        "fraude":            1,
    })
    return pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed)


def load_or_generate(csv_path=None, **kwargs) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    return generate_fraud_data(**kwargs)


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "data_sample" / "fraud_transactions_simulated.csv"
    df = generate_fraud_data()
    df.to_csv(out, index=False)
    print(f"Créé : {out} | {len(df):,} lignes | fraude : {df.fraude.mean():.3%}")
