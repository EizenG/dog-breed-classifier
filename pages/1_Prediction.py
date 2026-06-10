"""Page Prédiction — upload d'image et classification de race."""
import os

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(page_title="Prédiction", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .result-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .breed-name {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
    }
    .confidence-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .confidence-high   { color: #4ade80; font-weight: 700; font-size: 1.4rem; }
    .confidence-medium { color: #facc15; font-weight: 700; font-size: 1.4rem; }
    .confidence-low    { color: #f87171; font-weight: 700; font-size: 1.4rem; }
    #MainMenu, footer  { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Prédiction de race")
st.caption("Dépose une photo de chien pour identifier sa race.")

uploaded = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with col_result:
        with st.spinner("Analyse en cours…"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error(f"Impossible de joindre l'API ({API_URL}).")
                st.stop()

        if resp.status_code == 200:
            result = resp.json()
            breed = result["breed"].replace("_", " ").title()
            confidence = result["confidence"] * 100

            if confidence >= 80:
                conf_class = "confidence-high"
            elif confidence >= 50:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"

            st.markdown(
                f'<div class="result-card">'
                f'<p class="breed-name">{breed}</p>'
                f'<p class="confidence-label">Confiance : '
                f'<span class="{conf_class}">{confidence:.1f} %</span></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Graphe Top 3 ──────────────────────────────────────────────
            top3 = result["top_3"]
            names = [t["breed"].replace("_", " ").title() for t in top3]
            values = [t["confidence"] * 100 for t in top3]
            colors = ["#38bdf8", "#818cf8", "#a78bfa"]

            fig = go.Figure(go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.1f} %" for v in values],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    range=[0, max(values) * 1.25],
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#e2e8f0", size=14),
                    autorange="reversed",
                ),
                margin=dict(l=0, r=60, t=10, b=10),
                height=160,
                showlegend=False,
            )
            st.markdown("**Top 3**")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        elif resp.status_code == 503:
            st.error("Le modèle n'est pas encore chargé. Réessaie dans quelques secondes.")
        else:
            st.error(f"Erreur API ({resp.status_code}) : {resp.json().get('detail', resp.text)}")

else:
    st.markdown(
        "<div style='text-align:center; color:#475569; padding:4rem 0; font-size:1.1rem;'>"
        "📂 Aucune image sélectionnée"
        "</div>",
        unsafe_allow_html=True,
    )
