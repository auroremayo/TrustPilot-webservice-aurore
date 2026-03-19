"""
Injection du CSS global — thème sombre moderne (glassmorphism + gradient indigo/violet).
"""

import streamlit as st


def inject_css() -> None:
    st.markdown("""
<style>
/* ── Fond principal ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0D1117 0%, #161B27 60%, #0D1117 100%);
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B27 100%);
    border-right: 1px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

/* ── Texte ── */
html, body, [class*="css"] { color: #E6EDF3; }
h1, h2, h3, h4              { color: #E6EDF3 !important; }
p, li, span, label          { color: #C9D1D9 !important; }

/* ── Titres gradient ── */
.hero-title {
    font-size: 2.8rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(135deg, #6366F1, #A78BFA, #38BDF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .3rem;
}
.hero-subtitle { color: #8B949E !important; font-size: 1.05rem; }

/* ── Boutons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(99,102,241,.35) !important;
    transition: all .25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,.55) !important;
    background: linear-gradient(135deg, #7C3AED, #6366F1) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(22,27,39,.9) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    border-radius: 10px !important; color: #E6EDF3 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}

/* ── Onglets ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.03); border-radius: 12px;
    padding: 4px; gap: 4px; border: 1px solid rgba(99,102,241,.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; color: #8B949E !important;
    font-weight: 500; padding: .4rem 1.1rem; transition: all .2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,.4);
}

/* ── Métriques ── */
[data-testid="metric-container"] {
    background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.25);
    border-radius: 14px; padding: 1rem 1.2rem; backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] { color: #A78BFA !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8B949E !important; }
[data-testid="stMetricDelta"] { color: #34D399 !important; }

/* ── Cards ── */
.card {
    background: rgba(22,27,39,.8); border: 1px solid rgba(99,102,241,.2);
    border-radius: 16px; padding: 1.4rem 1.6rem; backdrop-filter: blur(10px);
    margin-bottom: 1rem; transition: border-color .2s, transform .2s;
}
.card:hover { border-color: rgba(99,102,241,.45); transform: translateY(-1px); }

/* ── Badges sentiment ── */
.badge-positif {
    display: inline-block;
    background: linear-gradient(135deg, #064E3B, #065F46);
    color: #34D399 !important; border: 1px solid #34D399;
    border-radius: 50px; padding: .35rem 1.2rem;
    font-weight: 700; font-size: 1.15rem;
}
.badge-negatif {
    display: inline-block;
    background: linear-gradient(135deg, #450A0A, #7F1D1D);
    color: #F87171 !important; border: 1px solid #F87171;
    border-radius: 50px; padding: .35rem 1.2rem;
    font-weight: 700; font-size: 1.15rem;
}
.badge-neutre {
    display: inline-block;
    background: linear-gradient(135deg, #451A03, #78350F);
    color: #FBBF24 !important; border: 1px solid #FBBF24;
    border-radius: 50px; padding: .35rem 1.2rem;
    font-weight: 700; font-size: 1.15rem;
}

/* ── Alertes ── */
.alert-normal  { background: rgba(16,185,129,.1);  border-left: 4px solid #10B981; border-radius: 8px; padding: .8rem 1.2rem; color: #6EE7B7 !important; }
.alert-warning { background: rgba(245,158,11,.1);  border-left: 4px solid #F59E0B; border-radius: 8px; padding: .8rem 1.2rem; color: #FDE68A !important; }
.alert-critical{ background: rgba(239,68,68,.1);   border-left: 4px solid #EF4444; border-radius: 8px; padding: .8rem 1.2rem; color: #FCA5A5 !important; }

/* ── Barre quota ── */
.quota-bar-bg   { background: rgba(255,255,255,.08); border-radius: 20px; height: 10px; overflow: hidden; margin-top: 6px; }
.quota-bar-fill { height: 100%; border-radius: 20px; transition: width .4s ease; }

/* ── Divers ── */
hr { border-color: rgba(99,102,241,.2) !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(99,102,241,.2); border-radius: 12px; overflow: hidden; }
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #6366F1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)
