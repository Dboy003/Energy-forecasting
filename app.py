
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Energy Forecasting Monitor",
    page_icon="⚡",
    layout="wide"
)

MARGE_IC = 697

# ============================================================
# Chargement des données et du modèle
# ============================================================

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "pjme_final.csv",
        index_col="Datetime",
        parse_dates=True
    )
    return df

@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent
    return joblib.load(BASE_DIR / "models" / "model_xgboost.pkl")

df    = load_data()
model = load_model()

# Split test
FEATURES = [col for col in df.columns if col != "PJME"]
test     = df[df.index >= "2018-01-01"]
X_test   = test[FEATURES]
y_test   = test["PJME"]

# Prédictions
y_pred = model.predict(X_test)

# Calcul des erreurs
erreurs = pd.DataFrame({
    "reel"   : y_test.values,
    "pred"   : y_pred,
    "erreur" : y_test.values - y_pred,
    "mape"   : np.abs((y_test.values - y_pred) / y_test.values) * 100,
    "lower"  : y_pred - MARGE_IC,
    "upper"  : y_pred + MARGE_IC,
}, index=y_test.index)

erreurs["heure"]  = erreurs.index.hour
erreurs["mois"]   = erreurs.index.month
erreurs["alerte"] = erreurs["mape"] > 3.0

# ============================================================
# Header
# ============================================================

st.title("⚡ Energy Forecasting : Dashboard de Monitoring")
st.markdown("**Modèle** : XGBoost | **Région** : PJM Est | **Période de test** : 2018")
st.divider()

# ============================================================
# KPIs
# ============================================================

col1, col2, col3, col4 = st.columns(4)

mape_global = erreurs["mape"].mean()
mae_global  = np.abs(erreurs["erreur"]).mean()
rmse_global = np.sqrt((erreurs["erreur"]**2).mean())
alertes     = erreurs["alerte"].sum()

col1.metric(
    label="MAPE Global",
    value=f"{mape_global:.2f}%",
    delta=f"{3.0 - mape_global:.2f}% vs objectif 3%"
)
col2.metric(
    label="MAE",
    value=f"{mae_global:,.0f} MW"
)
col3.metric(
    label="RMSE",
    value=f"{rmse_global:,.0f} MW"
)
col4.metric(
    label="Alertes (MAPE > 3%)",
    value=f"{alertes}",
    delta=f"{alertes/len(erreurs)*100:.1f}% des heures",
    delta_color="inverse"
)

st.divider()

# ============================================================
# Sélecteur de période
# ============================================================

st.subheader("📈 Prédictions vs Réel")

col_debut, col_fin = st.columns(2)
with col_debut:
    debut = st.date_input(
        "Date de début",
        value=pd.to_datetime("2018-01-01"),
        min_value=pd.to_datetime("2018-01-01"),
        max_value=pd.to_datetime("2018-08-03")
    )
with col_fin:
    fin = st.date_input(
        "Date de fin",
        value=pd.to_datetime("2018-01-14"),
        min_value=pd.to_datetime("2018-01-01"),
        max_value=pd.to_datetime("2018-08-03")
    )

# Filtrage
mask    = (erreurs.index >= str(debut)) & (erreurs.index <= str(fin))
df_plot = erreurs[mask]

# Graphique prédictions
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["reel"],
    name="Réel", line=dict(color="steelblue", width=2)
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot["pred"],
    name="Prédiction", line=dict(color="red", width=1.5, dash="dash")
))
fig.add_trace(go.Scatter(
    x=df_plot.index.tolist() + df_plot.index.tolist()[::-1],
    y=df_plot["upper"].tolist() + df_plot["lower"].tolist()[::-1],
    fill="toself", fillcolor="rgba(0,200,0,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="IC 95% (±697 MW)"
))

fig.update_layout(
    title="Consommation réelle vs prédite (MW)",
    xaxis_title="Date",
    yaxis_title="Consommation (MW)",
    hovermode="x unified",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Evolution du MAPE + Alertes
# ============================================================

st.subheader("📊 Évolution du MAPE & Alertes")

mape_daily = erreurs["mape"].resample("D").mean()
alertes_daily = erreurs["alerte"].resample("D").sum()

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=("MAPE journalier (%)", 
                                     "Nombre d'alertes par jour (MAPE > 3%)"))

fig2.add_trace(go.Scatter(
    x=mape_daily.index, y=mape_daily.values,
    name="MAPE", line=dict(color="darkorange", width=1.5)
), row=1, col=1)

fig2.add_hline(
    y=3, line_dash="dash", line_color="red",
    annotation_text="Objectif 3%", row=1, col=1
)

fig2.add_trace(go.Bar(
    x=alertes_daily.index, y=alertes_daily.values,
    name="Alertes", marker_color="red", opacity=0.7
), row=2, col=1)

fig2.update_layout(height=500, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ============================================================
# Analyse des erreurs
# ============================================================

st.subheader("🔍 Analyse des Erreurs")

col_h, col_m = st.columns(2)

with col_h:
    mape_heure = erreurs.groupby("heure")["mape"].mean()
    fig3 = px.bar(
        x=mape_heure.index,
        y=mape_heure.values,
        labels={"x": "Heure", "y": "MAPE (%)"},
        title="MAPE moyen par heure",
        color=mape_heure.values,
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_m:
    mois_labels = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr",
                   5:"Mai", 6:"Jun", 7:"Jul", 8:"Aoû"}
    mape_mois = erreurs.groupby("mois")["mape"].mean()
    fig4 = px.bar(
        x=[mois_labels[m] for m in mape_mois.index],
        y=mape_mois.values,
        labels={"x": "Mois", "y": "MAPE (%)"},
        title="MAPE moyen par mois",
        color=mape_mois.values,
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ============================================================
# Distribution des erreurs
# ============================================================

st.subheader("📉 Distribution des Erreurs")

fig5 = px.histogram(
    erreurs, x="erreur", nbins=50,
    title="Distribution des erreurs (MW)",
    labels={"erreur": "Erreur (MW)"},
    color_discrete_sequence=["steelblue"]
)
fig5.add_vline(x=0, line_dash="dash", line_color="red")
st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# Footer
# ============================================================

st.divider()
st.markdown("""
**Stack** : Python · XGBoost · FastAPI · Streamlit · Docker  
**API** : [energy-forecasting-api.onrender.com](https://energy-forecasting-api.onrender.com)  
**GitHub** : [Dboy003/energy-forecasting](https://github.com/Dboy003/energy-forecasting)
""")
