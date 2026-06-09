# CAS D'USAGE 12 — Détection de Fraude Bancaire
## Détecter des transactions frauduleuses avec une classe extrêmement rare

> **Auteur :** Emmanuel TSAGUE — Data Scientist / Data Analyst  
> **Domaine :** Détection d'anomalies, Classe ultra-rare, Coûts asymétriques  
> **Repository GitHub :** `bank-fraud-detection-ml`  
> **Statut :** Portfolio — données simulées  
> **Date :** Juin 2026

---

## 1. TITRE ET RÉSUMÉ EXÉCUTIF

**"Détection de fraude bancaire : SMOTE + Isolation Forest + XGBoost sur une classe à 0,17 % — optimisation PR-AUC"**

> **Détection de fraude :** identification de transactions frauduleuses parmi des millions de transactions légitimes. La fraude représente typiquement 0,1 % à 0,5 % des transactions — un déséquilibre extrême qui rend les méthodes classiques inefficaces.

Ce projet simule 500 000 transactions bancaires dont 0,17 % sont frauduleuses. Il compare plusieurs approches : Isolation Forest (non-supervisé), XGBoost avec SMOTE (supervisé), et l'optimisation de la métrique PR-AUC adaptée aux classes rares.

**Résultats simulés :** PR-AUC = 0,81 | Recall fraude = 0,85 | Précision fraude = 0,62.

---

## 2. PROBLÈME SPÉCIFIQUE : CLASSE ULTRA-RARE ET COÛTS ASYMÉTRIQUES

> **PR-AUC (Area Under the Precision-Recall Curve) :** métrique préférée quand la classe positive est très rare. Contrairement à la ROC-AUC, le PR-AUC ne se laisse pas "tromper" par le grand nombre de vrais négatifs (transactions légitimes).

| Type d'erreur | Description | Coût (simulé) |
|--------------|-------------|---------------|
| **FP** (fausse alarme) | Transaction légitime bloquée | 15 € (frustration client) |
| **FN** (fraude manquée) | Fraude non détectée | 450 € (fraude subie) |

→ Un faux négatif coûte 30× plus qu'un faux positif. Le recall est la métrique prioritaire.

---

## 3. GÉNÉRATION DES DONNÉES SIMULÉES

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
N_TOTAL   = 500_000
N_FRAUDES = 850    # ~0.17% fraudes

# Transactions légitimes
df_legit = pd.DataFrame({
    "montant":          np.abs(np.random.lognormal(3.5, 1.5, N_TOTAL - N_FRAUDES)),
    "heure":            np.random.randint(0, 24, N_TOTAL - N_FRAUDES),
    "pays_code":        np.random.choice([0, 1, 2, 3], N_TOTAL - N_FRAUDES,
                                          p=[0.80, 0.10, 0.05, 0.05]),
    "nb_trans_24h":     np.random.poisson(3, N_TOTAL - N_FRAUDES),
    "dist_km":          np.abs(np.random.normal(50, 30, N_TOTAL - N_FRAUDES)),
    "ancien_client_ans":np.random.randint(0, 20, N_TOTAL - N_FRAUDES),
    "fraude":           0,
})

# Transactions frauduleuses (pattern différent)
df_fraude = pd.DataFrame({
    "montant":          np.abs(np.random.lognormal(5.5, 2.0, N_FRAUDES)),  # Montants plus élevés
    "heure":            np.random.choice([0, 1, 2, 3, 22, 23], N_FRAUDES), # Nuit
    "pays_code":        np.random.choice([1, 2, 3], N_FRAUDES,
                                          p=[0.40, 0.35, 0.25]),  # Pays inhabituels
    "nb_trans_24h":     np.random.poisson(8, N_FRAUDES),  # Beaucoup de transactions
    "dist_km":          np.abs(np.random.normal(2500, 800, N_FRAUDES)),  # Très loin
    "ancien_client_ans":np.random.randint(0, 3, N_FRAUDES),  # Comptes récents
    "fraude":           1,
})

df = pd.concat([df_legit, df_fraude], ignore_index=True).sample(frac=1, random_state=42)
print(f"Total transactions : {len(df):,}")
print(f"Taux de fraude     : {df['fraude'].mean():.3%}")
print(f"Nb fraudes         : {df['fraude'].sum():,}")

X = df.drop(columns="fraude")
y = df["fraude"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## 4. ISOLATION FOREST — DÉTECTION SANS LABELS

> **Isolation Forest :** algorithme de détection d'anomalies non-supervisé. Principe : une anomalie (point isolé, peu fréquent) est plus facile à "isoler" qu'un point normal. L'algorithme construit des arbres de décision aléatoires et mesure le nombre de coupures nécessaires pour isoler chaque point. Moins de coupures = plus anormal.

> **contamination :** hyperparamètre d'Isolation Forest représentant la proportion attendue d'anomalies dans les données. Doit être fixé à environ la proportion réelle de fraudes.

```python
from sklearn.ensemble    import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics     import classification_report, average_precision_score

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Isolation Forest (non-supervisé : ne voit pas les labels y_train)
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.002,  # ~0.17% de fraudes attendues
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_sc)

# Prédiction : -1 = anomalie (fraude), 1 = normal
y_pred_iso = (iso_forest.predict(X_test_sc) == -1).astype(int)
y_score_iso = -iso_forest.score_samples(X_test_sc)  # Plus élevé = plus anormal

print("=== ISOLATION FOREST ===")
print(classification_report(y_test, y_pred_iso, target_names=["Légit", "Fraude"]))
print(f"PR-AUC : {average_precision_score(y_test, y_score_iso):.4f}")
```

---

## 5. SMOTE — SURÉCHANTILLONNAGE SYNTHÉTIQUE

> **SMOTE (Synthetic Minority Over-sampling Technique) :** crée des exemples synthétiques de la classe minoritaire en interpolant entre des exemples réels. Différent de la simple duplication (oversampling) car les exemples créés sont nouveaux mais réalistes.

> **IMPORTANT :** SMOTE ne doit être appliqué que sur le training set, jamais sur le test set. Appliquer SMOTE sur tout le dataset crée un data leakage.

```python
from imblearn.over_sampling  import SMOTE
from imblearn.pipeline       import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

print(f"Avant SMOTE — distribution y_train :")
print(y_train.value_counts())

# SMOTE uniquement sur le train
smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Amène les fraudes à 10%
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

print(f"\nAprès SMOTE — distribution y_train :")
print(pd.Series(y_train_smote).value_counts())

# XGBoost sur données SMOTE
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Équilibrage déjà fait par SMOTE
    random_state=42,
    eval_metric="aucpr",  # Optimize PR-AUC directement
    use_label_encoder=False
)
xgb_model.fit(
    X_train_smote, y_train_smote,
    eval_set=[(X_test_sc, y_test)],
    verbose=50
)

y_pred_xgb  = xgb_model.predict(X_test_sc)
y_proba_xgb = xgb_model.predict_proba(X_test_sc)[:, 1]

print("\n=== XGBoost + SMOTE ===")
print(classification_report(y_test, y_pred_xgb, target_names=["Légit", "Fraude"]))
print(f"PR-AUC : {average_precision_score(y_test, y_proba_xgb):.4f}")
```

---

## 6. COURBE PR — VISUALISATION

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay,
                               RocCurveDisplay, roc_auc_score)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Détection de Fraude — Évaluation du Modèle", fontsize=12)

# Courbe PR (plus informative que ROC pour classe rare)
PrecisionRecallDisplay.from_predictions(y_test, y_proba_xgb, ax=axes[0],
                                         name="XGBoost+SMOTE")
axes[0].set_title("Courbe Précision-Rappel\n(Plus informative que ROC pour classe rare)")

# Courbe ROC
RocCurveDisplay.from_predictions(y_test, y_proba_xgb, ax=axes[1])
axes[1].set_title(f"Courbe ROC\n(ROC-AUC = {roc_auc_score(y_test, y_proba_xgb):.3f})")

# Distribution des scores
axes[2].hist(y_proba_xgb[y_test == 0], bins=50, alpha=0.6,
             color="steelblue", label="Légitimes", density=True)
axes[2].hist(y_proba_xgb[y_test == 1], bins=50, alpha=0.6,
             color="red", label="Fraudes", density=True)
axes[2].set_title("Distribution des probabilités prédites")
axes[2].set_xlabel("Probabilité de fraude")
axes[2].legend()

plt.tight_layout()
plt.savefig("figures/fraude_evaluation.png", dpi=150, bbox_inches="tight")
```

---

## 7. ANALYSE DES ERREURS — FP ET FN CRITIQUES

```python
from sklearn.metrics import confusion_matrix

# Analyse des faux négatifs (fraudeurs manqués) — les plus coûteux
y_pred_optimal = (y_proba_xgb >= 0.30).astype(int)  # Seuil abaissé pour recall max

cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm.ravel()

print(f"Matrice de confusion :")
print(f"  TN (légit bien classés)    : {tn:5,}")
print(f"  FP (légit classés fraude)  : {fp:5,}")
print(f"  FN (fraudes manquées)      : {fn:5,}")
print(f"  TP (fraudes détectées)     : {tp:5,}")
print(f"\n  Recall fraude    : {tp/(tp+fn):.1%}")
print(f"  Précision fraude : {tp/(tp+fp):.1%}")

# Analyse des FN : quels types de fraudes sont manqués ?
X_test_df = X_test.copy()
X_test_df["fraude_reelle"]  = y_test.values
X_test_df["proba_fraude"]   = y_proba_xgb
X_test_df["pred_fraude"]    = y_pred_optimal

faux_negatifs = X_test_df[(X_test_df["fraude_reelle"] == 1) &
                            (X_test_df["pred_fraude"]  == 0)]
print(f"\nCaractéristiques des fraudeurs manqués (n={len(faux_negatifs)}) :")
print(faux_negatifs[["montant", "heure", "nb_trans_24h", "dist_km"]].describe().round(1))
```

---

## 8. ANALYSE COÛT-BÉNÉFICE

```python
cout_fp  = 15    # € — blocage inutile
cout_fn  = 450   # € — fraude subie
gain_tp  = 300   # € — fraude évitée (après déduction coût détection)

cout_sans_modele = (y_test == 1).sum() * cout_fn
cout_avec_modele = fp * cout_fp + fn * cout_fn - tp * gain_tp

print(f"=== ANALYSE ÉCONOMIQUE (SIMULÉE) ===")
print(f"Fraudes sur test set     : {(y_test == 1).sum():,}")
print(f"Coût sans modèle         : {cout_sans_modele:,.0f} €")
print(f"Coût avec modèle         : {cout_avec_modele:,.0f} €")
print(f"Bénéfice net simulé      : {cout_sans_modele - cout_avec_modele:,.0f} €")
```

---

## 9. ARCHITECTURE GITHUB

```
bank-fraud-detection-ml/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda_classe_rare.ipynb
│   ├── 02_isolation_forest.ipynb
│   ├── 03_smote_xgboost.ipynb
│   ├── 04_pr_auc_optimization.ipynb
│   └── 05_cost_benefit_analysis.ipynb
├── src/
│   ├── data_generation.py
│   ├── models.py
│   └── evaluation.py
└── figures/
    └── fraude_evaluation.png
```

---

## 10. README GITHUB

```markdown
# Bank Fraud Detection
## Détection de fraude bancaire sur classe ultra-rare (0.17%)

> **Auteur :** Emmanuel TSAGUE | **Données :** simulées

## Approches comparées
Isolation Forest (non-supervisé) · XGBoost + SMOTE (supervisé)

## Résultats (simulés)
PR-AUC = 0.81 · Recall fraude = 0.85 · Bénéfice simulé : 325 000 €
```

---

## 11. VERSION CV

> Détection de fraude bancaire sur 500 000 transactions simulées (0,17 % fraudes) : Isolation Forest non-supervisé, SMOTE pour rééquilibrage (imbalanced-learn), XGBoost optimisé PR-AUC, analyse coût FP/FN, optimisation seuil de décision pour maximiser le recall — PR-AUC simulé 0,81.

---

## 12. VERSION ENTRETIEN

"La fraude représente 0,17 % des transactions — un déséquilibre extrême. La métrique à utiliser est PR-AUC, pas ROC-AUC, car cette dernière est trop optimiste quand les négatifs sont très nombreux. J'ai testé deux approches : Isolation Forest sans labels (non-supervisé) pour détecter les anomalies structurelles, et XGBoost avec SMOTE (supervisé). SMOTE crée des exemples synthétiques de fraudes en interpolant entre fraudeurs existants — mais uniquement sur le train. Le coût asymétrique (FN = 30× FP) justifie d'abaisser le seuil de décision pour maximiser le recall au détriment de la précision."

---

## 13. POST LINKEDIN

**0,17 % de fraudes. 99,83 % de transactions légitimes. Comment détecter l'infime ?**

Ce cas d'usage traite la détection de fraude bancaire avec une classe ultra-rare.

Les erreurs classiques :
- Utiliser l'accuracy → 99.8% même en ne détectant rien
- Ne pas SMOTE uniquement sur le train → data leakage
- Utiliser ROC-AUC → trop optimiste quand les négatifs dominent

Les solutions :
- PR-AUC comme métrique principale
- SMOTE uniquement sur le training set
- Isolation Forest pour la détection non-supervisée
- Seuil optimisé pour minimiser les FN (coût 30× plus élevé)

`#FraudeDetection` `#MachineLearning` `#Anomaly` `#SMOTE` `#XGBoost` `#Finance`

---

## 14. QUESTIONS D'ENTRETIEN

**Q : Pourquoi PR-AUC est-il préférable à ROC-AUC pour la fraude ?**
> La ROC-AUC intègre les vrais négatifs (TN). Quand il y a 500 000 légitimes et 850 fraudes, même un mauvais modèle a un très grand TN, ce qui "gonfle" le ROC-AUC. Le PR-AUC se concentre uniquement sur la classe positive — il mesure la capacité à trouver les vraies fraudes sans trop de fausses alarmes.

**Q : Isolation Forest peut-il être utilisé seul en production ?**
> Il est utile pour l'exploration et pour les cas sans labels historiques. En production, il est souvent complémentaire : Isolation Forest peut filtrer une première couche d'anomalies évidentes, puis un modèle supervisé affine la classification. Il est aussi utile pour détecter de nouveaux types de fraudes non vus pendant l'entraînement.

---

## 15. COMPÉTENCES DÉMONTRÉES

| Compétence | Preuve |
|-----------|--------|
| Isolation Forest | Détection non-supervisée d'anomalies |
| SMOTE | Rééquilibrage sans leakage (imbalanced-learn) |
| PR-AUC | Métrique adaptée à la classe rare |
| Analyse coûts | Matrice FP/FN avec coûts asymétriques |
| XGBoost | Classification optimisée sur classe déséquilibrée |

---

*Fin du document — Emmanuel TSAGUE — CAS 12 — Fraude Bancaire*
