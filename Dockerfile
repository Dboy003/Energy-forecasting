FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY data/processed/model_xgboost.pkl ./data/processed/
COPY data/processed/scaler_X.pkl ./data/processed/
COPY data/processed/scaler_y.pkl ./data/processed/

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
