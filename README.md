# ⚡ Prévision de Consommation Énergétique du PJM Est

> Prédire la demande électrique à H+1 avec **MAPE 0.74%** sur données réelles PJM (2003-2018)

[![API Status](https://img.shields.io/badge/API-Online-green)](https://energy-forecasting-api.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)

---

## 📋 Contexte

Le gestionnaire de réseau électrique PJM doit équilibrer en permanence production et consommation. Une erreur de prévision coûte **100,000€/heure** de déséquilibre réseau.

**Objectif** : prédire la consommation horaire avec MAPE < 3%

**Résultat obtenu** : MAPE **0.74%** sur le set de test (2018)

---

## 🏗️ Architecture du projet

```text
energy-forecasting/
├── notebooks/
│   ├── 01_exploration.ipynb          # Analyse et nettoyage des données
│   ├── 02_feature_engineering.ipynb  # 35 features (temporelles, météo, lags)
│   ├── 03_baseline_models.ipynb      # Modèles de référence
│   ├── 04_advanced_models.ipynb      # XGBoost + LSTM
│   ├── 05_evaluation.ipynb           # Évaluation finale + intervalles de confiance
│   └── 06_api.ipynb                  # API FastAPI + README
├── api/
│   └── main.py                       # Code de l'API
├── models/                           # Modèles entraînés
├── data/
│   ├── raw/                          # Données brutes (non versionnées)
│   └── processed/                    # Données traitées
├── app.py                            # Dashboard Streamlit
└── Dockerfile
```
---

## 📊 Dataset

- **Source** : [Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Région** : PJM Est (Philadelphia, PA)
- **Période** : 2003 → 2018 (145,366 heures)
- **Météo** : Open-Meteo API (température, humidité, vent, couverture nuageuse)

---

## 🔧 Feature Engineering

35 features construites en 5 catégories :

| Catégorie | Features | Exemple |
|-----------|----------|---------|
| Temporelles | 8 features | hour, dayofweek, month, season... |
| Lags | 5 features | lag_1, lag_24, lag_168, lag_8736 |
| Rolling | 5 features | rolling_mean_24h, rolling_std_168h... |
| Météo | 4 features | temp, humidity, windspeed, cloudcover |
| Cycliques & interactions | 9 features | hour_sin/cos, temp_x_season... |

---

## 🤖 Modèles & Résultats

| Modèle | MAPE Validation | MAPE Test |
|--------|----------------|-----------|
| Moyenne Mobile | 11.71% | — |
| Naïf Saisonnier | 6.78% | — |
| Régression Linéaire | 2.55% | — |
| LSTM | 1.75% | 1.72% |
| **XGBoost** | **0.65%** | **0.74%** ✅ |

**Objectif initial : MAPE < 3% → Résultat : 0.74%** 🎯

---

## 📏 Intervalles de Confiance

Trois méthodes testées et comparées :

| Méthode | Taux de couverture | Verdict |
|---------|-------------------|---------|
| Bootstrap | 41.1% | ❌ |
| Quantile Regression | 84.6% | ❌ |
| **Conformal Prediction** | **95.1%** | ✅ |

Marge finale : **±697 MW** autour de chaque prédiction

---

## 🚀 API

L'API est déployée sur Render et accessible publiquement :

**Base URL** : `https://energy-forecasting-api.onrender.com`

> ⚠️ Première requête : 30-60 secondes (cold start Render free tier)

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Statut de l'API |
| `/health` | GET | Santé du modèle |
| `/predict` | POST | Prédiction H+1 + intervalle de confiance |

### Exemple de requête

```bash
curl -X POST "https://energy-forecasting-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "datetime": "2018-01-15T14:00:00",
    "lag_1": 35000,
    "lag_24": 34500,
    "lag_48": 34200,
    "lag_168": 33800,
    "lag_8736": 34100,
    "rolling_mean_24h": 33500,
    "rolling_mean_168h": 33200,
    "rolling_mean_720h": 32800,
    "rolling_std_24h": 2100,
    "rolling_std_168h": 2300,
    "temp": 2.5,
    "humidity": 75,
    "windspeed": 15,
    "cloudcover": 60
  }'
```

### Exemple de réponse

```json
{
  "datetime": "2018-01-15T14:00:00",
  "prediction_mw": 35420.5,
  "confidence_interval": {
    "lower": 34723.5,
    "upper": 36117.5
  },
  "mape_expected_percent": 0.74
}
```

### Documentation interactive

👉 [https://energy-forecasting-api.onrender.com/docs](https://energy-forecasting-api.onrender.com/docs)

---

## 🛠️ Installation locale

```bash
# Cloner le repo
git clone https://github.com/Dboy003/energy-forecasting.git
cd energy-forecasting

# Créer l'environnement virtuel
py -3.11 -m venv venv
.\venv\Scripts\Activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api.main:app --reload --port 8000
```

---

## 📚 Stack technique

| Catégorie | Outils |
|-----------|--------|
| Langage | Python 3.11 |
| ML | XGBoost, Scikit-learn, TensorFlow/Keras |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn |
| Déploiement | Docker, Render |
| Versioning | Git, GitHub |
| Météo | Open-Meteo API |

---

## 👤 Auteur

**Mourad** — [GitHub](https://github.com/Dboy003)