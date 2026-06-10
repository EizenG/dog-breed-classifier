"""Streamlit — Dog Breed Classifier (page d'accueil)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Header gradient */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { color: #e2e8f0; font-size: 2.8rem; margin: 0; font-weight: 700; }
    .hero p  { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }

    /* Status badges */
    .badge-ok  { background:#065f46; color:#6ee7b7; padding:6px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }
    .badge-err { background:#7f1d1d; color:#fca5a5; padding:6px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }

    /* Breed pill */
    .breed-pill {
        display:inline-block;
        background:#1e3a5f;
        color:#93c5fd;
        border:1px solid #2563eb44;
        padding:4px 12px;
        border-radius:20px;
        font-size:0.85rem;
        margin:3px;
    }

    /* Stat card */
    .stat-card {
        background:#1e293b;
        border:1px solid #334155;
        border-radius:12px;
        padding:1.2rem;
        text-align:center;
    }
    .stat-value { font-size:2rem; font-weight:700; color:#38bdf8; }
    .stat-label { font-size:0.85rem; color:#94a3b8; margin-top:4px; }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🐕 Dog Breed Classifier</h1>
    <p>Classification de races de chiens · Stanford Dogs Dataset · InceptionV3 · 97.9 % accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Statut API ────────────────────────────────────────────────────────────────
col_status, col_breeds = st.columns([1, 2], gap="large")

with col_status:
    st.markdown("#### Statut de l'API")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        if resp.status_code == 200 and data.get("model_loaded"):
            st.markdown('<span class="badge-ok">● API connectée — modèle prêt</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-err">⚠ API connectée — modèle non chargé</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="badge-err">✕ API inaccessible</span>', unsafe_allow_html=True)
        st.caption(f"URL configurée : `{API_URL}`")
        st.stop()

# ── Races disponibles ─────────────────────────────────────────────────────────
with col_breeds:
    try:
        resp = requests.get(f"{API_URL}/breeds", timeout=5)
        breeds = resp.json()["breeds"]
        st.markdown("#### Races reconnues")
        pills = "".join(
            f'<span class="breed-pill">{b.replace("_", " ").title()}</span>'
            for b in breeds
        )
        st.markdown(pills, unsafe_allow_html=True)
    except Exception:
        pass

st.divider()

# ── Stats ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("10", "Races"),
    ("97.9 %", "Accuracy (InceptionV3)"),
    ("96.3 %", "Accuracy (EfficientNetB0)"),
    ("3", "Modèles entraînés"),
]
for col, (val, label) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    "<p style='color:#64748b; text-align:center; font-size:0.85rem;'>"
    "Utilise le menu à gauche pour accéder à la <b>Prédiction</b> ou au <b>Monitoring</b>."
    "</p>",
    unsafe_allow_html=True,
)
