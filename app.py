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

FEATURES = [col for col in df.columns if col != "PJME"]
test     = df[df.index >= "2018-01-01"]
X_test   = test[FEATURES]
y_test   = test["PJME"]

y_pred = model.predict(X_test)

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
# Fonction prévision itérative
# ============================================================

@st.cache_data
def prevision_iterative(n_heures=24):
    historique     = df.copy()
    derniere_heure = df.index[-1]
    predictions    = []
    timestamps     = []

    for h in range(1, n_heures + 1):
        ts_pred      = derniere_heure + pd.Timedelta(hours=h)
        derniere_ligne = historique.iloc[-1].copy()

        features = {
            "is_outlier"       : 0,
            "hour"             : ts_pred.hour,
            "dayofweek"        : ts_pred.dayofweek,
            "month"            : ts_pred.month,
            "quarter"          : ts_pred.quarter,
            "year"             : ts_pred.year,
            "dayofyear"        : ts_pred.dayofyear,
            "weekofyear"       : ts_pred.isocalendar()[1],
            "season"           : (ts_pred.month % 12) // 3,
            "is_holiday"       : 0,
            "is_weekend"       : int(ts_pred.dayofweek >= 5),
            "lag_1"            : historique["PJME"].iloc[-1],
            "lag_24"           : historique["PJME"].iloc[-24] if len(historique) >= 24 else historique["PJME"].iloc[0],
            "lag_48"           : historique["PJME"].iloc[-48] if len(historique) >= 48 else historique["PJME"].iloc[0],
            "lag_168"          : historique["PJME"].iloc[-168] if len(historique) >= 168 else historique["PJME"].iloc[0],
            "lag_8736"         : historique["PJME"].iloc[-8736] if len(historique) >= 8736 else historique["PJME"].iloc[0],
            "rolling_mean_24h" : historique["PJME"].iloc[-24:].mean(),
            "rolling_mean_168h": historique["PJME"].iloc[-168:].mean(),
            "rolling_mean_720h": historique["PJME"].iloc[-720:].mean(),
            "rolling_std_24h"  : historique["PJME"].iloc[-24:].std(),
            "rolling_std_168h" : historique["PJME"].iloc[-168:].std(),
            "temp"             : derniere_ligne["temp"],
            "humidity"         : derniere_ligne["humidity"],
            "windspeed"        : derniere_ligne["windspeed"],
            "cloudcover"       : derniere_ligne["cloudcover"],
            "hour_sin"         : np.sin(2 * np.pi * ts_pred.hour / 24),
            "hour_cos"         : np.cos(2 * np.pi * ts_pred.hour / 24),
            "dow_sin"          : np.sin(2 * np.pi * ts_pred.dayofweek / 7),
            "dow_cos"          : np.cos(2 * np.pi * ts_pred.dayofweek / 7),
            "month_sin"        : np.sin(2 * np.pi * ts_pred.month / 12),
            "month_cos"        : np.cos(2 * np.pi * ts_pred.month / 12),
            "temp_x_hour"      : derniere_ligne["temp"] * ts_pred.hour,
            "temp_x_season"    : derniere_ligne["temp"] * ((ts_pred.month % 12) // 3),
            "hour_x_weekend"   : ts_pred.hour * int(ts_pred.dayofweek >= 5),
        }

        X    = pd.DataFrame([features])
        pred = float(model.predict(X)[0])
        predictions.append(pred)
        timestamps.append(ts_pred)

        nouvelle_ligne         = derniere_ligne.copy()
        nouvelle_ligne["PJME"] = pred
        nouvelle_ligne.name    = ts_pred
        historique = pd.concat([historique, nouvelle_ligne.to_frame().T])

    return timestamps, predictions

# ============================================================
# Navigation
# ============================================================

st.sidebar.title("⚡ Energy Forecasting")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Monitoring", "🔮 Prévision H+24"]
)

# ============================================================
# PAGE 1 : Monitoring
# ============================================================

if page == "📊 Monitoring":
    st.title("📊 Dashboard de Monitoring")
    st.markdown("**Modèle** : XGBoost | **Région** : PJM Est | **Période de test** : 2018")
    st.divider()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    mape_global = erreurs["mape"].mean()
    mae_global  = np.abs(erreurs["erreur"]).mean()
    rmse_global = np.sqrt((erreurs["erreur"]**2).mean())
    alertes     = erreurs["alerte"].sum()

    col1.metric("MAPE Global", f"{mape_global:.2f}%",
                f"{3.0 - mape_global:.2f}% vs objectif 3%")
    col2.metric("MAE", f"{mae_global:,.0f} MW")
    col3.metric("RMSE", f"{rmse_global:,.0f} MW")
    col4.metric("Alertes (MAPE > 3%)", f"{alertes}",
                f"{alertes/len(erreurs)*100:.1f}% des heures",
                delta_color="inverse")

    st.divider()

    # Sélecteur de période
    st.subheader("📈 Prédictions vs Réel")
    col_debut, col_fin = st.columns(2)
    with col_debut:
        debut = st.date_input("Date de début",
                              value=pd.to_datetime("2018-01-01"),
                              min_value=pd.to_datetime("2018-01-01"),
                              max_value=pd.to_datetime("2018-08-03"))
    with col_fin:
        fin = st.date_input("Date de fin",
                            value=pd.to_datetime("2018-01-14"),
                            min_value=pd.to_datetime("2018-01-01"),
                            max_value=pd.to_datetime("2018-08-03"))

    mask    = (erreurs.index >= str(debut)) & (erreurs.index <= str(fin))
    df_plot = erreurs[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["reel"],
                             name="Réel", line=dict(color="steelblue", width=2)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["pred"],
                             name="Prédiction",
                             line=dict(color="red", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(
        x=df_plot.index.tolist() + df_plot.index.tolist()[::-1],
        y=df_plot["upper"].tolist() + df_plot["lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(0,200,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="IC 95% (±697 MW)"
    ))
    fig.update_layout(title="Consommation réelle vs prédite (MW)",
                      xaxis_title="Date", yaxis_title="Consommation (MW)",
                      hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Evolution MAPE + Alertes
    st.subheader("📊 Évolution du MAPE & Alertes")
    mape_daily    = erreurs["mape"].resample("D").mean()
    alertes_daily = erreurs["alerte"].resample("D").sum()

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("MAPE journalier (%)",
                                         "Nombre d'alertes par jour (MAPE > 3%)"))
    fig2.add_trace(go.Scatter(x=mape_daily.index, y=mape_daily.values,
                              name="MAPE",
                              line=dict(color="darkorange", width=1.5)),
                   row=1, col=1)
    fig2.add_hline(y=3, line_dash="dash", line_color="red",
                   annotation_text="Objectif 3%", row=1, col=1)
    fig2.add_trace(go.Bar(x=alertes_daily.index, y=alertes_daily.values,
                          name="Alertes", marker_color="red", opacity=0.7),
                   row=2, col=1)
    fig2.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Analyse erreurs
    st.subheader("🔍 Analyse des Erreurs")
    col_h, col_m = st.columns(2)

    with col_h:
        mape_heure = erreurs.groupby("heure")["mape"].mean()
        fig3 = px.bar(x=mape_heure.index, y=mape_heure.values,
                      labels={"x": "Heure", "y": "MAPE (%)"},
                      title="MAPE moyen par heure",
                      color=mape_heure.values,
                      color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig3, use_container_width=True)

    with col_m:
        mois_labels   = {1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr",
                         5:"Mai", 6:"Jun", 7:"Jul", 8:"Aoû"}
        mape_mois     = erreurs.groupby("mois")["mape"].mean()
        fig4 = px.bar(x=[mois_labels[m] for m in mape_mois.index],
                      y=mape_mois.values,
                      labels={"x": "Mois", "y": "MAPE (%)"},
                      title="MAPE moyen par mois",
                      color=mape_mois.values,
                      color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # Distribution erreurs
    st.subheader("📉 Distribution des Erreurs")
    fig5 = px.histogram(erreurs, x="erreur", nbins=50,
                        title="Distribution des erreurs (MW)",
                        labels={"erreur": "Erreur (MW)"},
                        color_discrete_sequence=["steelblue"])
    fig5.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig5, use_container_width=True)

# ============================================================
# PAGE 2 : Prévision H+24
# ============================================================

elif page == "🔮 Prévision H+24":
    st.title("🔮 Prévision Itérative H+24")
    st.markdown("""
    Prévision de la consommation pour les **24 prochaines heures** 
    au-delà des données disponibles.
    
    Chaque prédiction devient l'input de la suivante.
    """)
    st.divider()

    # Bouton de lancement
    if st.button("🚀 Lancer la prévision H+24", type="primary"):
        with st.spinner("Calcul des prévisions en cours..."):
            timestamps, predictions = prevision_iterative(24)

        st.success("✅ Prévisions calculées !")

        # KPIs prévision
        col1, col2, col3 = st.columns(3)
        col1.metric("Horizon", "24 heures")
        col2.metric("Prévision min", f"{min(predictions):,.0f} MW")
        col3.metric("Prévision max", f"{max(predictions):,.0f} MW")

        st.divider()

        # Graphique
        derniere_heure = df.index[-1]
        dernieres_48h  = df["PJME"].iloc[-48:]

        fig = go.Figure()

        # Historique
        fig.add_trace(go.Scatter(
            x=dernieres_48h.index, y=dernieres_48h.values,
            name="Historique connu",
            line=dict(color="steelblue", width=2)
        ))

        # Prédictions
        fig.add_trace(go.Scatter(
            x=timestamps, y=predictions,
            name="Prévision H+1 à H+24",
            line=dict(color="red", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=6)
        ))

        # Intervalle de confiance
        fig.add_trace(go.Scatter(
            x=timestamps + timestamps[::-1],
            y=[p + MARGE_IC for p in predictions] +
              [p - MARGE_IC for p in predictions[::-1]],
            fill="toself", fillcolor="rgba(255,0,0,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"IC 95% (±{MARGE_IC} MW)"
        ))

        # Ligne séparation
        fig.add_vline(x=derniere_heure.timestamp() * 1000,
              line_dash="dash", line_color="green",
              annotation_text="Fin données connues")

        fig.update_layout(
            title="Prévision itérative H+24 — Au-delà des données disponibles",
            xaxis_title="Date",
            yaxis_title="Consommation (MW)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Tableau
        st.subheader("📋 Tableau des prévisions")
        df_prev = pd.DataFrame({
            "Horizon"       : [f"H+{i+1}" for i in range(24)],
            "Datetime"      : [str(ts) for ts in timestamps],
            "Prévision (MW)": [round(p) for p in predictions],
            "IC Bas (MW)"   : [round(p - MARGE_IC) for p in predictions],
            "IC Haut (MW)"  : [round(p + MARGE_IC) for p in predictions],
        })
        st.dataframe(df_prev, use_container_width=True)

    else:
        st.info("👆 Clique sur le bouton pour lancer la prévision H+24")

# ============================================================
# Footer
# ============================================================

st.divider()
st.markdown("""
**Stack** : Python · XGBoost · FastAPI · Streamlit · Docker  
**API** : [energy-forecasting-api.onrender.com](https://energy-forecasting-api.onrender.com)  
**GitHub** : [Dboy003/energy-forecasting](https://github.com/Dboy003/energy-forecasting)
""")