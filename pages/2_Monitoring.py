"""Page Monitoring — détection de data drift via Evidently."""
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .status-ok  {
        background:#064e3b; border:1px solid #065f46;
        border-radius:12px; padding:1rem 1.4rem;
        color:#6ee7b7; font-weight:600; font-size:1rem;
    }
    .status-err {
        background:#450a0a; border:1px solid #7f1d1d;
        border-radius:12px; padding:1rem 1.4rem;
        color:#fca5a5; font-weight:600; font-size:1rem;
    }
    .metric-card {
        background:#1e293b; border:1px solid #334155;
        border-radius:12px; padding:1.2rem;
        text-align:center;
    }
    .metric-value { font-size:2rem; font-weight:700; color:#38bdf8; }
    .metric-drift { font-size:2rem; font-weight:700; color:#f87171; }
    .metric-label { font-size:0.82rem; color:#94a3b8; margin-top:4px; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Monitoring — Data Drift")

# ── Bouton analyse ─────────────────────────────────────────────────────────────
col_hdr, col_btn = st.columns([5, 1])
with col_btn:
    if st.button("🔄 Analyser", type="primary", use_container_width=True):
        with st.spinner("Calcul Evidently en cours…"):
            try:
                resp = requests.post(
                    f"{API_URL}/drift/run",
                    params={"min_rows": 5},
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                st.error(f"API inaccessible ({API_URL}).")
                st.stop()
        if resp.status_code == 200:
            st.success("Analyse terminée.")
            st.rerun()
        else:
            st.error(f"Erreur ({resp.status_code}) : {resp.json().get('detail', resp.text)}")

# ── Chargement du dernier résumé ───────────────────────────────────────────────
try:
    resp = requests.get(f"{API_URL}/drift/summary", timeout=10)
except requests.exceptions.ConnectionError:
    st.error(f"API inaccessible ({API_URL}).")
    st.stop()

if resp.status_code == 404:
    st.info("Aucun rapport disponible. Clique sur **Analyser** une fois que l'API a reçu des requêtes.")
    st.stop()

if resp.status_code != 200:
    st.error(f"Erreur API ({resp.status_code}).")
    st.stop()

summary = resp.json()
drift_detected = summary["dataset_drift_detected"]

# ── Statut global ──────────────────────────────────────────────────────────────
status_html = (
    '<div class="status-err">⚠️ Drift détecté — la distribution de production s\'écarte du référentiel.</div>'
    if drift_detected else
    '<div class="status-ok">✅ Pas de drift — la distribution de production est stable.</div>'
)
st.markdown(status_html, unsafe_allow_html=True)
st.caption(
    f"Dernière analyse : {summary['timestamp'][:19].replace('T', ' ')} "
    f"· {summary['production_rows']} requêtes analysées"
)

st.divider()

# ── Métriques ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
cards = [
    (str(summary["total_features"]), "metric-value", "Features analysées"),
    (str(summary["drifted_features"]), "metric-drift" if summary["drifted_features"] else "metric-value", "Features en drift"),
    (str(summary["production_rows"]), "metric-value", "Requêtes production"),
]
for col, (val, cls, label) in zip([c1, c2, c3], cards):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="{cls}">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Graphe p-values ────────────────────────────────────────────────────────────
st.subheader("p-values par feature")
st.caption("Seuil de détection : p < 0.05")

features = list(summary["per_feature"].keys())
pvalues = [summary["per_feature"][f]["p_value"] for f in features]
drifted = [summary["per_feature"][f]["drift_detected"] for f in features]
bar_colors = ["#f87171" if d else "#4ade80" for d in drifted]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=features,
    y=pvalues,
    marker=dict(color=bar_colors, line=dict(width=0)),
    text=[f"{p:.3f}" for p in pvalues],
    textposition="outside",
    textfont=dict(color="#e2e8f0"),
))
fig.add_hline(
    y=0.05,
    line=dict(color="#facc15", width=2, dash="dash"),
    annotation_text="seuil 0.05",
    annotation_font_color="#facc15",
    annotation_position="top right",
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(tickfont=dict(color="#e2e8f0"), showgrid=False),
    yaxis=dict(
        tickfont=dict(color="#94a3b8"),
        gridcolor="#1e293b",
        range=[0, max(pvalues + [0.12])],
        title=dict(text="p-value", font=dict(color="#94a3b8")),
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=300,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Tableau détaillé ──────────────────────────────────────────────────────────
st.subheader("Détail par feature")
rows_html = ""
for feature, stats in summary["per_feature"].items():
    status = "⚠️ Drift" if stats["drift_detected"] else "✅ Stable"
    row_bg = "background:#450a0a" if stats["drift_detected"] else ""
    rows_html += (
        f"<tr style='{row_bg}'>"
        f"<td style='padding:8px 12px;color:#e2e8f0'>{feature}</td>"
        f"<td style='padding:8px 12px'>{status}</td>"
        f"<td style='padding:8px 12px;color:#94a3b8'>{stats['stattest']}</td>"
        f"<td style='padding:8px 12px;color:#38bdf8;font-weight:600'>{stats['p_value']:.4f}</td>"
        f"</tr>"
    )

st.markdown(
    f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
        <thead>
            <tr style='background:#1e293b;color:#94a3b8;text-align:left;'>
                <th style='padding:10px 12px'>Feature</th>
                <th style='padding:10px 12px'>Statut</th>
                <th style='padding:10px 12px'>Test</th>
                <th style='padding:10px 12px'>p-value</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """,
    unsafe_allow_html=True,
)
