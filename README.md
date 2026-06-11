# Détection de Fraude Bancaire — Classe Ultra-Rare

> **SMOTE + Isolation Forest + XGBoost sur une classe à 0,17 % — optimisation PR-AUC**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Domaine](https://img.shields.io/badge/Domaine-Finance-green)
![Statut](https://img.shields.io/badge/Statut-Portfolio-orange)
![Données](https://img.shields.io/badge/Données-Simulées%2FAnonymisées-lightgrey)

---

## Contexte métier

La fraude bancaire représente moins de 0,2 % des transactions — un déséquilibre extrême qui rend les approches classiques inefficaces. Les coûts sont asymétriques : un FN (fraude manquée) coûte 30× plus qu'un FP (fausse alarme).

---

## Problème traité

500 000 transactions simulées avec 0,17 % de fraudes. Détecter les fraudes avec un recall maximal, minimiser les coûts métier et justifier le choix de PR-AUC vs ROC-AUC.

---

## Solution proposée

Isolation Forest non-supervisé pour la détection d'anomalies structurelles, SMOTE uniquement sur le train (prévention leakage), XGBoost optimisé PR-AUC, seuil de décision optimisé pour minimiser les FN.

---

## Technologies utilisées

| Outil | Usage |
|-------|-------|
| Python 3.10+ | Langage principal |
| pandas / numpy | Manipulation des données |
| scikit-learn | Machine Learning & preprocessing |
| matplotlib / seaborn | Visualisation |
| Jupyter Notebook | Exploration interactive |

> Voir `requirements.txt` pour la liste complète.

---

## Structure du projet

```
bank-fraud-detection-ml/
├── README.md              ← Ce fichier
├── PORTFOLIO.md           ← Documentation complète du cas d'usage
├── .gitignore
├── requirements.txt
├── notebooks/             ← Jupyter Notebooks d'exploration
├── src/                   ← Code Python modulaire
├── data_sample/           ← Données simulées (anonymisées)
├── figures/               ← Graphiques et visualisations
├── reports/               ← Rapports et synthèses
└── docs/                  ← Documentation complémentaire
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TSAGUE25/bank-fraud-detection-ml.git
cd bank-fraud-detection-ml

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook
```

---

## Métriques clés (données simulées)

```
PR-AUC = 0.81 | Recall fraude = 0.85 | Précision fraude = 0.62 (simulés)
```

---

## Valeur métier

Bénéfice simulé de 325 000 € vs absence de détection automatisée.

---

## Limites

SMOTE crée des exemples synthétiques — peut sur-représenter des patterns bruités.

---

## Prochaines améliorations

AutoEncoder pour détection non-supervisée. Monitoring dérive concept. Graph Neural Networks.

---

## Avertissement — Confidentialité

> **Toutes les données utilisées dans ce projet sont simulées, synthétiques ou anonymisées.**
> Aucune donnée réelle, confidentielle ou propriétaire n'est présente dans ce dépôt.
> Ce projet est un cas d'usage pédagogique à destination du portfolio professionnel d'Emmanuel TSAGUE.

---

## Contributors

**TSAGUE EMMANUEL** - Data Scientist  
Specialise en Machine Learning, Data Analysis et systemes decisionnels.  
Formation Datascientest 2024 | EDF MAD EDVANCE  
Email : [emmatsague@yahoo.fr](mailto:emmatsague@yahoo.fr)  
GitHub : [github.com/TSAGUE25](https://github.com/TSAGUE25)

