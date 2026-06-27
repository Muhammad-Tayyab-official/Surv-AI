"""
app.py  —  SurvAI  |  Micro-Business Survival Predictor
Deep Navy + Gold professional theme

STATISTICS CONCEPTS WOVEN INTO THIS FILE:
  • Central Tendency       — mean/median of scores shown on Analytics page
  • Measure of Dispersion  — std dev, IQR, CV shown in population stats
  • Box Plot               — score distributions for Survived vs Failed
  • Probability Intro      — prior and posterior shown on results card
  • Contingency Table      — chi-squared independence check (Analytics)
  • Conditional Probability — P(Survived | answer) heatmap
  • Multiplicative Rule    — explained in methodology page
  • Law of Total Prob      — explained in methodology page
  • Bayes' Theorem         — the core engine (model.py + explained here)
  • Random Variable        — each question answer is a discrete RV
  • Probability Distribution — answer distributions shown in analytics
  • Binomial Distribution  — calibration check uses binomial logic
  • Poisson Distribution   — models rare "zero-sale day" events (Analytics)
  • Normal Distribution    — population score curves shown on results
  • Confidence Interval    — Wilson 95% CI shown on every prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import os
from datetime import datetime

from model import SurvivalPredictor, save_user_response, save_feedback, get_data_stats

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SurvAI | Micro-Business Resilience",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════
PRIMARY   = "#1E3A5F"
GOLD      = "#B8860B"
GOLD_LT   = "#F59E0B"
SUCCESS   = "#059669"
WARNING   = "#D97706"
DANGER    = "#DC2626"
TEXT      = "#0F172A"
MUTED     = "#475569"
BORDER    = "#CBD5E1"
BG        = "#F1F5F9"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
}}
.stApp {{ background-color: {BG}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0F172A 0%, {PRIMARY} 100%);
}}
[data-testid="stSidebar"] * {{ color: #F1F5F9 !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.1); }}
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}}

/* ── Cards ── */
.card {{
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}}
.card:hover {{ box-shadow: 0 4px 14px rgba(0,0,0,0.09); }}

/* ── Section label ── */
.label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {PRIMARY};
    padding-bottom: 6px;
    border-bottom: 2px solid {GOLD};
    display: inline-block;
    margin-bottom: 14px;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #2563EB 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    width: 100%;
}}
.stButton > button:hover {{
    box-shadow: 0 4px 18px rgba(30,58,95,0.35);
    transform: translateY(-1px);
}}

/* ── Data rows ── */
.drow {{
    display: flex;
    justify-content: space-between;
    padding: 11px 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.93rem;
}}
.drow:last-child {{ border-bottom: none; }}
.dlabel {{ color: {MUTED}; font-weight: 500; }}
.dval   {{ font-weight: 700; color: {PRIMARY}; }}

/* ── Status badges ── */
.badge-good {{ display:inline-block; padding:7px 15px; border-radius:999px;
    font-size:0.82rem; font-weight:700; background:rgba(5,150,105,0.1);
    color:{SUCCESS}; border:1px solid rgba(5,150,105,0.25); }}
.badge-warn {{ display:inline-block; padding:7px 15px; border-radius:999px;
    font-size:0.82rem; font-weight:700; background:rgba(217,119,6,0.1);
    color:{WARNING}; border:1px solid rgba(217,119,6,0.25); }}
.badge-bad  {{ display:inline-block; padding:7px 15px; border-radius:999px;
    font-size:0.82rem; font-weight:700; background:rgba(220,38,38,0.1);
    color:{DANGER};  border:1px solid rgba(220,38,38,0.25); }}

/* ── Metrics ── */
[data-testid="stMetricValue"] {{
    font-family:'Inter',sans-serif !important;
    font-weight:800 !important;
    color:{TEXT} !important;
    font-size:1.7rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-weight:600 !important;
    color:{MUTED} !important;
    font-size:0.82rem !important;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    border:1px solid {BORDER} !important;
    border-radius:8px !important;
}}

footer {{ visibility:hidden; }}
h1,h2,h3,h4,h5,h6 {{ color:{TEXT} !important; font-weight:700 !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# QUESTION DATA
# ══════════════════════════════════════════════════════════════════
QUESTIONS = {
    "Group A — Business Stability": {
        "q1":  {"q": "How long have you been buying from your main supplier?",
                "opts": ["Less than 3 months", "3–12 months", "More than 12 months"],
                "why": "Longer supplier relationship = trust, credit flexibility, reliable stock"},
        "q2":  {"q": "How many different suppliers do you rely on?",
                "opts": ["Only 1", "2–3 suppliers", "4 or more suppliers"],
                "why": "2–3 is ideal. Too few = risky. Too many = no deep relationships"},
        "q3":  {"q": "Do you sell from the same location every day?",
                "opts": ["Mobile — different places each day", "Semi-fixed (same area, not same spot)", "Yes — fixed stall or shop"],
                "why": "Fixed location lets customers find you again"},
        "q4":  {"q": "How many days per week do you operate?",
                "opts": ["1–2 days", "3–4 days", "5–7 days"],
                "why": "More open days = more customer contact"},
        "q5":  {"q": "Do customers come to you, or do you go find them?",
                "opts": ["Mostly I go find them", "Mixed", "Mostly they come to me"],
                "why": "Customers coming to you is more stable than chasing them"},
    },
    "Group B — Financial Buffer": {
        "q6":  {"q": "If your business earned nothing for a while, how many months could your household survive?",
                "opts": ["0–1 month", "1–3 months", "More than 3 months"],
                "why": "This is the single strongest predictor of survival"},
        "q7":  {"q": "Does anyone else in your household earn income?",
                "opts": ["No — I am the only earner", "Yes — one other person", "Yes — two or more others"],
                "why": "Multiple earners reduce pressure on this business"},
        "q8":  {"q": "How much does your daily income vary in a normal week?",
                "opts": ["Very unpredictable (can double or halve)", "Some variation (20–50%)", "Mostly stable (within 20%)"],
                "why": "Wild income swings make planning impossible"},
        "q9":  {"q": "Do you set aside money specifically for restocking your business?",
                "opts": ["No — I use whatever I have that day", "Sometimes", "Yes — I do this regularly"],
                "why": "Keeping business money separate from household money is key"},
    },
    "Group C — Demand & Competition": {
        "q10": {"q": "In the last month, how many days did you have almost no customers?",
                "opts": ["7 or more days", "3–6 days", "0–2 days"],
                "why": "Frequent empty days signal a demand problem"},
        "q11": {"q": "How many other sellers nearby offer the same thing as you?",
                "opts": ["5 or more", "2–4", "0–1"],
                "why": "Too many identical vendors splits the customer base"},
        "q12": {"q": "Do your customers come back regularly or are they mostly one-time?",
                "opts": ["Mostly random one-time buyers", "Mix of regulars and new", "Mostly the same regulars"],
                "why": "Repeat customers are predictable income"},
    },
    "Group D — Personal Drive & Growth": {
        "q13": {"q": "Have you tried anything new in the last 3 months? (new product, location, pricing…)",
                "opts": ["No — kept things exactly the same", "Yes — tried something, but it didn't work", "Yes — tried something and it worked"],
                "why": "Willingness to experiment separates survivors from those who freeze"},
        "q14": {"q": "Do you keep any record of sales, expenses, or stock?",
                "opts": ["No — I don't track", "Yes — I keep a mental track", "Yes — I use a written notebook or phone"],
                "why": "Even basic tracking shows intentional management"},
        "q15": {"q": "Looking 6 months ahead, do you think your business will grow, stay the same, or shrink?",
                "opts": ["Shrink or I am unsure", "Stay the same", "Grow"],
                "why": "Your own outlook often reveals information only you know"},
    },
}

VIZ_LABELS = {
    "q1":"Supplier Relation","q2":"Supplier Diversity","q3":"Location Type",
    "q4":"Operating Days","q5":"Customer Source","q6":"Savings Buffer",
    "q7":"Family Income","q8":"Income Stability","q9":"Cash Discipline",
    "q10":"Customer Demand","q11":"Competition Level","q12":"Customer Loyalty",
    "q13":"Innovation","q14":"Record Keeping","q15":"Future Outlook",
}

ADVICE = {
    "q1":"Build a longer relationship with your main supplier — loyalty brings better prices and credit.",
    "q2":"Aim for 2–3 reliable suppliers rather than depending on one or juggling too many.",
    "q3":"Try to secure a fixed spot. Customers return when they know where to find you.",
    "q4":"Try to open more days per week. Every extra day is more customer touchpoints.",
    "q5":"Build a spot that draws customers to you — consistency and visibility help.",
    "q6":"Start saving even a small emergency fund. One month of buffer changes everything.",
    "q7":"Explore ways for another household member to earn income so less pressure sits on this business.",
    "q8":"Identify what causes your income swings (weather? season?) and plan around them.",
    "q9":"Separate your business cash from your personal money. Set aside restocking funds first.",
    "q10":"Ask your customers why they are not coming. Adapt your product, price, or location.",
    "q11":"Find one way to stand out — better service, unique product, or a more convenient spot.",
    "q12":"Start remembering regular customers by name. Loyalty is built one person at a time.",
    "q13":"Try one small experiment this week — a new item, a different display, or a price test.",
    "q14":"Start writing down your daily sales total. Even one number per day reveals patterns.",
    "q15":"Set one small, achievable goal for this month and track progress weekly.",
}

# ══════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════
PFONT = dict(family="Inter, sans-serif", color=TEXT)
MFONT = dict(family="Inter, sans-serif", color=MUTED)

def _layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=10, l=10, r=10),
        legend=dict(font=dict(family="Inter", size=11, color=MUTED),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def chart_gauge(prob, prior):
    """Speedometer showing P(Survived). Delta vs. prior (base rate)."""
    bar_col = SUCCESS if prob >= 60 else (WARNING if prob >= 35 else DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={"reference": prior, "increasing": {"color": SUCCESS}, "decreasing": {"color": DANGER}},
        number={"font": {"family":"Inter","color":TEXT,"size":46,"weight":"bold"}, "suffix":"%"},
        gauge={
            "axis": {"range":[0,100], "tickcolor":BORDER, "tickfont":MFONT, "tickwidth":1},
            "bar":  {"color": bar_col, "thickness": 0.30},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range":[0,35],  "color":"rgba(220,38,38,0.07)"},
                {"range":[35,60], "color":"rgba(217,119,6,0.07)"},
                {"range":[60,100],"color":"rgba(5,150,105,0.07)"},
            ],
            "threshold": {"line":{"color":TEXT,"width":3},"thickness":0.75,"value":prob},
        },
    ))
    fig.update_layout(height=270, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=20,b=10,l=30,r=30))
    return fig


def chart_normal_curves(score):
    """
    NORMAL DISTRIBUTION — Population score curves.

    We model total resilience scores as normally distributed:
      Survived ~ N(μ=34, σ=4)
      Failed   ~ N(μ=21, σ=5.5)

    These parameters are estimated from mock data.
    The vertical line shows where THIS vendor's score falls.
    Overlap between the two curves = the difficulty of prediction.
    """
    x  = np.linspace(8, 52, 300)
    sy = stats.norm.pdf(x, 34, 4)     # Survived distribution
    fy = stats.norm.pdf(x, 21, 5.5)   # Failed distribution

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=fy, fill="tozeroy",
        fillcolor="rgba(220,38,38,0.13)", line=dict(color=DANGER, width=2.5), name="Failed"))
    fig.add_trace(go.Scatter(x=x, y=sy, fill="tozeroy",
        fillcolor="rgba(5,150,105,0.13)", line=dict(color=SUCCESS, width=2.5), name="Survived"))

    peak_y = max(stats.norm.pdf(score, 34, 4), stats.norm.pdf(score, 21, 5.5)) * 1.25
    fig.add_trace(go.Scatter(x=[score, score], y=[0, peak_y], mode="lines",
        line=dict(color=PRIMARY, width=2.5, dash="dash"), name="Your Score"))

    fig.update_xaxes(title_text="Resilience Score (out of 45)", gridcolor="rgba(0,0,0,0.05)",
                     tickfont=PFONT, title_font=dict(family="Inter", size=12, color=MUTED))
    fig.update_yaxes(showticklabels=False, zeroline=False)
    return _layout(fig)


def chart_boxplot(score):
    """
    BOX PLOT — Score distribution comparison.

    A box plot shows:
      - Median (middle line)
      - IQR = Q3 − Q1 (the box)  ← Measure of Dispersion
      - Whiskers = min/max within 1.5×IQR
      - The diamond = THIS vendor's score

    CENTRAL TENDENCY: median of Survived ≈ 34, Failed ≈ 21
    DISPERSION: the spread of the box shows variability within each group
    """
    np.random.seed(99)
    s_scores = np.random.normal(34, 4,   120).clip(18, 45)
    f_scores = np.random.normal(21, 5.5, 120).clip(10, 38)

    fig = go.Figure()
    fig.add_trace(go.Box(y=f_scores, name="Failed",   marker_color=DANGER,
        boxmean="sd", width=0.4, line=dict(color=DANGER,   width=1.5)))
    fig.add_trace(go.Box(y=s_scores, name="Survived", marker_color=SUCCESS,
        boxmean="sd", width=0.4, line=dict(color=SUCCESS,  width=1.5)))
    fig.add_trace(go.Scatter(x=["Failed","Survived"], y=[score, score], mode="markers",
        marker=dict(size=16, color=PRIMARY, symbol="diamond",
                    line=dict(color="white", width=2.5)),
        name=f"Your Score: {score}"))

    fig.update_yaxes(title_text="Resilience Score", gridcolor="rgba(0,0,0,0.05)",
                     tickfont=PFONT, zeroline=False)
    fig.update_xaxes(tickfont=PFONT)
    return _layout(fig)


def chart_waterfall(result):
    """
    Feature contribution waterfall.
    Each bar = (likelihood ratio − 1) for that question.
    Positive = helps survival, Negative = hurts survival.
    """
    qs   = list(result["question_impact"].keys())
    labs = [VIZ_LABELS.get(q, q) for q in qs]
    vals = [result["question_impact"][q] - 1 for q in qs]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(qs),
        x=labs, y=vals,
        connector={"line": {"color": BORDER, "width": 1}},
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        textfont=dict(family="Inter", size=9, color=MUTED),
        decreasing={"marker": {"color": DANGER,  "line": {"color": DANGER,  "width": 0}}},
        increasing={"marker": {"color": SUCCESS, "line": {"color": SUCCESS, "width": 0}}},
    ))
    fig.update_xaxes(tickfont=dict(family="Inter", size=9, color=TEXT), tickangle=-45)
    fig.update_yaxes(title_text="Impact (ratio − 1)", gridcolor="rgba(0,0,0,0.05)",
                     zeroline=True, zerolinecolor=BORDER, tickfont=PFONT)
    return _layout(fig, height=400)


def chart_radar(result):
    """Radar chart of 6 key dimensions."""
    keys = ["q1","q6","q8","q12","q10","q15"]
    cats = ["Supplier<br>Relation","Savings<br>Buffer","Income<br>Stability",
            "Customer<br>Loyalty","Customer<br>Demand","Future<br>Outlook"]
    vals = [result["question_impact"].get(k, 1.0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor=f"rgba(30,58,95,0.13)", line=dict(color=PRIMARY, width=2.5), name="Your Profile"))
    fig.add_trace(go.Scatterpolar(r=[1]*6, theta=cats, fill="none",
        line=dict(color="#94A3B8", width=1.5, dash="dash"), name="Baseline"))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(vals)*1.15, 1.35)],
                            showticklabels=False, gridcolor="rgba(0,0,0,0.07)"),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.07)",
                             tickfont=dict(family="Inter", size=11, color=TEXT)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(font=dict(family="Inter", size=11, color=MUTED),
                    orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        height=310, margin=dict(t=40,b=20,l=40,r=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# INITIALISE MODEL
# ══════════════════════════════════════════════════════════════════
predictor = SurvivalPredictor()
sd_init   = get_data_stats()
if sd_init["real_users"] > 0:
    predictor.train_model()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 20px 0; text-align:center;">
  <div style="display:flex;align-items:center;justify-content:center;gap:10px;margin-bottom:8px;">
    <div style="width:42px;height:42px;background:linear-gradient(135deg,#1E3A5F,#2563EB);
         border-radius:10px;display:flex;align-items:center;justify-content:center;
         font-size:1rem;font-weight:800;color:white;">S:AI</div>
    <h2 style="margin:0;font-size:1.6rem;font-weight:800;color:#F1F5F9 !important;">SurvAI</h2>
  </div>
  <div style="width:48px;height:2px;background:linear-gradient(90deg,#B8860B,#F59E0B);
       margin:4px auto;border-radius:2px;"></div>
  <p style="font-size:0.7rem;margin-top:6px;opacity:0.5;letter-spacing:0.06em;color:#F1F5F9;">
    RESILIENCE INTELLIGENCE
  </p>
</div>
""", unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["📋 Assessment", "📊 Analytics", "🗄️ Database", "📚 Methodology"],
                    label_visibility="collapsed")

    st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.45;margin-bottom:10px;">System Status</p>', unsafe_allow_html=True)

    sd = get_data_stats()
    st.markdown(f"""
<div style="margin-bottom:14px;">
  <div style="font-size:0.72rem;opacity:0.55;">Baseline Records</div>
  <div style="font-size:1.2rem;font-weight:700;">{sd['mock_vendors']}</div>
</div>
<div style="margin-bottom:14px;">
  <div style="font-size:0.72rem;opacity:0.55;">Assessments Run</div>
  <div style="font-size:1.2rem;font-weight:700;">{sd['real_users']}</div>
</div>
<div style="margin-bottom:20px;">
  <div style="font-size:0.72rem;opacity:0.55;">Verified Outcomes</div>
  <div style="font-size:1.2rem;font-weight:700;">{sd['with_feedback']}</div>
</div>
""", unsafe_allow_html=True)

    if st.button("🔄 Retrain Model"):
        predictor.train_model()
        st.success("Model retrained.")
        st.rerun()

    st.markdown("""
<div style="margin-top:36px;font-size:0.62rem;opacity:0.3;text-align:center;">
SurvAI v3.0 · Bayesian Engine · 15 Concepts
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE — ASSESSMENT
# ══════════════════════════════════════════════════════════════════
if page == "📋 Assessment":

    st.markdown("""
<div style="margin-bottom:28px;">
  <h1 style="font-size:2.1rem;margin-bottom:4px;">New Assessment</h1>
  <div style="width:54px;height:3px;background:#B8860B;border-radius:2px;margin-bottom:14px;"></div>
  <p style="color:#475569;font-size:1rem;max-width:680px;margin:0;">
    Answer 15 simple questions. No financial documents needed.
    The Bayesian engine will estimate your 18-month survival probability.
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", "15")
    c2.metric("Time Needed", "< 5 min")
    c3.metric("Training Records", predictor.model["n_total"])
    c4.metric("Confidence Level", "95%")

    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

    if "show_results" not in st.session_state:
        st.session_state.show_results     = False
        st.session_state.prediction_result = None

    with st.form("assessment_form"):
        answers = {}
        for section, qs in QUESTIONS.items():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="label">{section}</div>', unsafe_allow_html=True)
            for qk, qd in qs.items():
                st.markdown(f"""
<div style="font-weight:600;color:{TEXT};margin:16px 0 6px 0;font-size:0.94rem;">
  {qd['q']}
</div>
<div style="font-size:0.8rem;color:{MUTED};margin-bottom:6px;">{qd['why']}</div>
""", unsafe_allow_html=True)
                sel = st.selectbox(f"Answer {qk}", qd["opts"], key=f"s_{qk}",
                                   label_visibility="collapsed")
                answers[qk] = qd["opts"].index(sel) + 1   # convert to 1/2/3

            st.markdown('</div>', unsafe_allow_html=True)

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            submitted = st.form_submit_button("🔍 Generate Prediction", use_container_width=True)

        if submitted:
            with st.spinner("Running Bayesian inference…"):
                result = predictor.predict(answers)
                ts     = sum(answers.values())
                vid    = save_user_response(answers, ts, result)
                st.session_state.prediction_result = result
                st.session_state.total_score       = ts
                st.session_state.vendor_id         = vid
                st.session_state.show_results      = True
            st.rerun()

    # ── RESULTS ──────────────────────────────────────────────────
    if st.session_state.show_results and st.session_state.prediction_result:
        r  = st.session_state.prediction_result
        ts = st.session_state.total_score

        st.markdown("<hr style='margin:36px 0;border-color:#E2E8F0;'>", unsafe_allow_html=True)
        st.markdown('<h2 style="margin-bottom:20px;">Your Assessment Results</h2>', unsafe_allow_html=True)

        # ── Top result card + gauge ──
        rc1, rc2 = st.columns([1.1, 2])
        with rc1:
            badge = (("badge-good","Low Risk") if r["probability"] >= 60 else
                     ("badge-warn","Medium Risk") if r["probability"] >= 35 else
                     ("badge-bad","High Risk"))
            chg_col = SUCCESS if r["change"] > 0 else DANGER

            st.markdown(f"""
<div class="card" style="height:100%;">
  <div style="font-size:0.82rem;color:{MUTED};font-weight:600;text-transform:uppercase;
       letter-spacing:0.05em;margin-bottom:4px;">Survival Probability</div>
  <div style="font-size:3.4rem;font-weight:800;color:{TEXT};line-height:1.1;
       margin-bottom:10px;">{r['probability']}%</div>
  <span class="{badge[0]}" style="margin-bottom:18px;display:inline-block;">{badge[1]}</span>

  <div class="drow">
    <span class="dlabel">95% Confidence Interval</span>
    <span class="dval">{r['ci_lower']}% – {r['ci_upper']}%</span>
  </div>
  <div class="drow">
    <span class="dlabel">Prior (base rate)</span>
    <span class="dval">{r['prior']}%</span>
  </div>
  <div class="drow">
    <span class="dlabel">Model Adjustment</span>
    <span class="dval" style="color:{chg_col};">{r['change']:+.1f}%</span>
  </div>
  <div class="drow">
    <span class="dlabel">Resilience Score</span>
    <span class="dval">{ts} / 45</span>
  </div>
  <div class="drow">
    <span class="dlabel">Age Modifier</span>
    <span class="dval">×{r['age_modifier']}</span>
  </div>
  <div class="drow" style="border-bottom:none;">
    <span class="dlabel">Reference ID</span>
    <span style="font-family:monospace;font-size:0.75rem;color:{MUTED};">
      {st.session_state.vendor_id}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        with rc2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label">Bayesian Posterior Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_gauge(r["probability"], r["prior"]), use_container_width=True)
            st.markdown(f"""
<div style="font-size:0.82rem;color:{MUTED};margin-top:-8px;line-height:1.6;">
  <b>How to read this:</b> The needle shows your posterior probability.
  The delta (▲/▼) shows how much your answers shifted the 30% base rate.
  The three shaded zones are High Risk (red), Medium Risk (amber), Low Risk (green).
</div>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Strengths & Weaknesses ──
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="label" style="color:{SUCCESS};border-bottom-color:{SUCCESS};">✅ Positive Signals</div>', unsafe_allow_html=True)
            if r["strengths"]:
                for s in r["strengths"]:
                    st.markdown(f"""
<div style="display:flex;align-items:flex-start;margin-bottom:12px;">
  <div style="width:8px;height:8px;background:{SUCCESS};border-radius:50%;
       margin:5px 12px 0 0;flex-shrink:0;"></div>
  <div>
    <div style="font-weight:700;font-size:0.93rem;">{VIZ_LABELS.get(s,s)}</div>
    <div style="font-size:0.82rem;color:{MUTED};">This factor is boosting your survival odds.</div>
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:{MUTED};font-size:0.9rem;'>No standout strengths above baseline.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="label" style="color:{DANGER};border-bottom-color:{DANGER};">⚠️ Risk Factors</div>', unsafe_allow_html=True)
            if r["weaknesses"]:
                for w in r["weaknesses"]:
                    st.markdown(f"""
<div style="display:flex;align-items:flex-start;margin-bottom:12px;">
  <div style="width:8px;height:8px;background:{DANGER};border-radius:3px;
       margin:5px 12px 0 0;flex-shrink:0;"></div>
  <div>
    <div style="font-weight:700;font-size:0.93rem;">{VIZ_LABELS.get(w,w)}</div>
    <div style="font-size:0.82rem;color:{MUTED};">{ADVICE.get(w,'Work on improving this area.')}</div>
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:{MUTED};font-size:0.9rem;'>No critical weaknesses flagged.</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Charts row 1 ──
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label">Dimensional Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_radar(r), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label">Feature Contribution (Waterfall)</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_waterfall(r), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Charts row 2 — Box Plot + Normal Distribution ──
        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label">Box Plot — Score Distribution</div>', unsafe_allow_html=True)
            st.caption("Your diamond vs. the typical range for survived vs. failed businesses.")
            st.plotly_chart(chart_boxplot(ts), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cr:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label">Normal Distribution — Population Curves</div>', unsafe_allow_html=True)
            st.caption("The dashed line shows where your score falls in each population.")
            st.plotly_chart(chart_normal_curves(ts), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Outcome Logging ──
        st.markdown(f"""
<div class="card" style="background:#F8FAFC;border:1px dashed {BORDER};box-shadow:none;">
  <div class="label">📝 Outcome Logging</div>
  <p style="font-size:0.88rem;color:{MUTED};margin-bottom:14px;">
    Come back in 18 months and log whether this business survived.
    Every verified outcome retrains the model and improves future predictions.
  </p>
""", unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            fb = st.selectbox("Status", ["Status Unknown","Verified: Survived","Verified: Failed"],
                              key="fb", label_visibility="collapsed")
        with fc2:
            if st.button("Log Outcome", use_container_width=True):
                if "Survived" in fb:
                    save_feedback(st.session_state.vendor_id, r["probability"], "Survived")
                    predictor.train_model()
                    st.success("Logged ✓")
                elif "Failed" in fb:
                    save_feedback(st.session_state.vendor_id, r["probability"], "Failed")
                    predictor.train_model()
                    st.success("Logged ✓")
        with fc3:
            st.caption(f"Ref: {st.session_state.vendor_id}")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("""
<div style="margin-bottom:28px;">
  <h1 style="font-size:2.1rem;margin-bottom:4px;">Statistical Analytics</h1>
  <div style="width:54px;height:3px;background:#B8860B;border-radius:2px;margin-bottom:14px;"></div>
  <p style="color:#475569;font-size:1rem;max-width:680px;margin:0;">
    Deep-dive into all 15 statistics concepts powering SurvAI.
  </p>
</div>
""", unsafe_allow_html=True)

    df = pd.read_csv("data/mock_data.csv")
    survived = df[df["survival_outcome"] == "Survived"]["total_score"]
    failed   = df[df["survival_outcome"] == "Failed"]["total_score"]

    # ── CENTRAL TENDENCY + DISPERSION TABLE ──────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📐 Central Tendency & Measure of Dispersion</div>', unsafe_allow_html=True)
    st.markdown(f"""
<p style="font-size:0.88rem;color:{MUTED};margin-bottom:14px;">
  <b>Central Tendency</b> = where the "middle" of a dataset is (mean, median, mode).<br>
  <b>Dispersion</b> = how spread out the data is (std dev, IQR, range, CV).
  A high CV means scores vary a lot; a low CV means they cluster tightly.
</p>
""", unsafe_allow_html=True)

    rows = [
        ("Mean (average)",          f"{survived.mean():.1f}", f"{failed.mean():.1f}"),
        ("Median (middle value)",   f"{survived.median():.1f}", f"{failed.median():.1f}"),
        ("Mode (most common)",      f"{survived.mode()[0]:.0f}", f"{failed.mode()[0]:.0f}"),
        ("Std Dev (spread)",        f"{survived.std():.1f}", f"{failed.std():.1f}"),
        ("Variance",                f"{survived.var():.1f}", f"{failed.var():.1f}"),
        ("IQR (Q3 − Q1)",           f"{stats.iqr(survived):.1f}", f"{stats.iqr(failed):.1f}"),
        ("Range (max − min)",       f"{survived.max()-survived.min():.0f}", f"{failed.max()-failed.min():.0f}"),
        ("CV % (std/mean × 100)",   f"{survived.std()/survived.mean()*100:.1f}%", f"{failed.std()/failed.mean()*100:.1f}%"),
        ("Skewness",                f"{stats.skew(survived):.2f}", f"{stats.skew(failed):.2f}"),
        ("Kurtosis",                f"{stats.kurtosis(survived):.2f}", f"{stats.kurtosis(failed):.2f}"),
    ]
    header_html = f"""
<div style="display:grid;grid-template-columns:2fr 1fr 1fr;font-size:0.78rem;
     font-weight:700;text-transform:uppercase;letter-spacing:0.05em;
     color:{MUTED};padding:8px 0;border-bottom:2px solid {BORDER};margin-bottom:4px;">
  <span>Statistic</span><span style="color:{SUCCESS};">▲ Survived</span>
  <span style="color:{DANGER};">▼ Failed</span>
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)
    for label, sv, fv in rows:
        st.markdown(f"""
<div style="display:grid;grid-template-columns:2fr 1fr 1fr;font-size:0.9rem;
     padding:9px 0;border-bottom:1px solid #F1F5F9;">
  <span style="color:{TEXT};font-weight:500;">{label}</span>
  <span style="font-weight:700;color:{SUCCESS};">{sv}</span>
  <span style="font-weight:700;color:{DANGER};">{fv}</span>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── BOX PLOT (population overview) ───────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📦 Box Plot — Full Population</div>', unsafe_allow_html=True)
    st.caption("The box = IQR (middle 50% of scores). Line inside = median. Whiskers = full range.")

    fig = go.Figure()
    fig.add_trace(go.Box(y=failed,   name="Failed",   marker_color=DANGER,
        boxmean="sd", line=dict(color=DANGER,   width=1.5)))
    fig.add_trace(go.Box(y=survived, name="Survived", marker_color=SUCCESS,
        boxmean="sd", line=dict(color=SUCCESS,  width=1.5)))
    fig.update_yaxes(title_text="Total Resilience Score", gridcolor="rgba(0,0,0,0.05)", tickfont=PFONT)
    fig.update_xaxes(tickfont=PFONT)
    st.plotly_chart(_layout(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── POISSON DISTRIBUTION ─────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🔢 Poisson Distribution — Zero-Sale Days</div>', unsafe_allow_html=True)
    st.markdown(f"""
<p style="font-size:0.88rem;color:{MUTED};margin-bottom:14px;">
  Q10 asks about "almost-empty days" per month.
  These rare events fit the <b>Poisson Distribution</b> — it models how often
  a rare event (zero-sale day) happens in a fixed time window.
  <br>Formula: P(X=k) = (λᵏ × e⁻λ) / k!  &nbsp; where λ = average events per period.
</p>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        lambda_val = st.slider("Set average zero-sale days per month (λ)", 0.5, 8.0, 2.0, 0.5)
    with col2:
        st.markdown(f"<br><p style='color:{MUTED};font-size:0.88rem;'>λ = {lambda_val:.1f} means on average {lambda_val:.1f} bad days per month.</p>", unsafe_allow_html=True)

    k_vals = np.arange(0, 15)
    p_vals = stats.poisson.pmf(k_vals, lambda_val)

    fig = go.Figure(go.Bar(x=k_vals, y=p_vals,
        marker_color=[SUCCESS if k <= 2 else (WARNING if k <= 6 else DANGER) for k in k_vals],
        text=[f"{v:.3f}" for v in p_vals], textposition="outside",
        textfont=dict(family="Inter", size=9, color=MUTED),
    ))
    fig.update_xaxes(title_text="Number of Zero-Sale Days (k)", tickfont=PFONT,
                     title_font=dict(family="Inter", size=12, color=MUTED))
    fig.update_yaxes(title_text="Probability P(X=k)", gridcolor="rgba(0,0,0,0.05)", tickfont=PFONT)
    st.plotly_chart(_layout(fig, height=300), use_container_width=True)
    st.markdown(f"""
<div style="font-size:0.83rem;color:{MUTED};margin-top:4px;">
  P(0 bad days) = {stats.poisson.pmf(0, lambda_val):.3f} &nbsp;|&nbsp;
  P(≤2 bad days) = {stats.poisson.cdf(2, lambda_val):.3f} &nbsp;|&nbsp;
  P(≥7 bad days) = {1 - stats.poisson.cdf(6, lambda_val):.3f}
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── BINOMIAL DISTRIBUTION ─────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🎲 Binomial Distribution — Vendor Survival Simulation</div>', unsafe_allow_html=True)
    st.markdown(f"""
<p style="font-size:0.88rem;color:{MUTED};margin-bottom:14px;">
  If we survey <b>N</b> vendors and each one has probability <b>p</b> of surviving,
  the number that actually survive follows a <b>Binomial(N, p)</b> distribution.
  <br>Formula: P(X=k) = C(N,k) × pᵏ × (1−p)^(N−k)
</p>
""", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    with bc1:
        n_vendors = st.slider("Number of vendors (N)", 10, 200, 50, 10)
    with bc2:
        p_surv = st.slider("Survival probability (p)", 0.10, 0.80, 0.30, 0.05)

    k_range  = np.arange(0, n_vendors + 1)
    binom_pmf = stats.binom.pmf(k_range, n_vendors, p_surv)
    expected  = n_vendors * p_surv
    std_binom = np.sqrt(n_vendors * p_surv * (1 - p_surv))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=k_range, y=binom_pmf, name="P(X=k)",
        marker_color=PRIMARY, opacity=0.7,
        marker=dict(line=dict(color="white", width=0.3))))
    fig.add_vline(x=expected, line_dash="dash", line_color=GOLD, line_width=2,
                  annotation_text=f"Expected = {expected:.0f}", annotation_font_color=GOLD)
    fig.update_xaxes(title_text="Number of Survivors (k)", tickfont=PFONT,
                     title_font=dict(family="Inter", size=12, color=MUTED))
    fig.update_yaxes(title_text="Probability", gridcolor="rgba(0,0,0,0.05)", tickfont=PFONT)
    st.plotly_chart(_layout(fig, height=300), use_container_width=True)
    st.markdown(f"""
<div style="font-size:0.83rem;color:{MUTED};">
  Expected survivors: <b>{expected:.0f}</b> &nbsp;|&nbsp;
  Std Dev: <b>±{std_binom:.1f}</b> &nbsp;|&nbsp;
  P(at least half survive): <b>{1-stats.binom.cdf(n_vendors//2-1, n_vendors, p_surv):.3f}</b>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE ────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📈 Feature Importance — Predictive Gap</div>', unsafe_allow_html=True)
    st.caption("Gap = P(Survived | best answer) − P(Survived | worst answer). Higher gap = more important question.")

    qstats = predictor.get_question_stats()
    df_q   = pd.DataFrame(qstats)
    df_q["display"] = df_q["question"].map(VIZ_LABELS)

    fig = go.Figure(go.Bar(
        y=df_q.sort_values("gap")["display"],
        x=df_q.sort_values("gap")["gap"],
        orientation="h",
        marker_color=PRIMARY,
        text=df_q.sort_values("gap")["gap"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(family="Inter", size=10, color=MUTED),
    ))
    fig.update_xaxes(title_text="Predictive Gap (%)", gridcolor="rgba(0,0,0,0.05)",
                     title_font=dict(family="Inter", size=12, color=MUTED), tickfont=MFONT)
    fig.update_yaxes(tickfont=dict(family="Inter", size=11, color=TEXT))
    st.plotly_chart(_layout(fig, height=480), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CONDITIONAL PROBABILITY HEATMAP ──────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🔥 Conditional Probability — P(Survived | Answer)</div>', unsafe_allow_html=True)
    st.caption("Each cell = probability of survival given that specific answer. Red = bad, Green = good.")

    q_order = [q["question"] for q in qstats]
    heatmap_vals = []
    for q in q_order:
        row = [
            predictor.model["conditional"][q].get("1", 0) * 100,
            predictor.model["conditional"][q].get("2", 0) * 100,
            predictor.model["conditional"][q].get("3", 0) * 100,
        ]
        heatmap_vals.append(row)

    y_labels = [VIZ_LABELS.get(q, q) for q in q_order]
    fig = go.Figure(go.Heatmap(
        z=heatmap_vals, x=["Weak Answer (1)","Medium Answer (2)","Strong Answer (3)"],
        y=y_labels,
        colorscale=[[0,"#FEE2E2"],[0.5,"#FEFCE8"],[1,"#D1FAE5"]],
        text=[[f"{v:.1f}%" for v in row] for row in heatmap_vals],
        texttemplate="%{text}", textfont={"size":11,"family":"Inter","color":TEXT},
        showscale=False,
    ))
    fig.update_xaxes(tickfont=dict(family="Inter", size=11, color=MUTED), side="top")
    fig.update_yaxes(tickfont=dict(family="Inter", size=11, color=TEXT))
    st.plotly_chart(_layout(fig, height=500), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CONTINGENCY TABLE ──────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📋 Contingency Table — Independence Check</div>', unsafe_allow_html=True)
    st.markdown(f"""
<p style="font-size:0.88rem;color:{MUTED};margin-bottom:14px;">
  A <b>contingency table</b> cross-tabulates two categorical variables.
  The chi-squared test checks if they are <b>independent</b>.
  Naive Bayes assumes all 15 questions are independent — violations here are documented limitations.
</p>
""", unsafe_allow_html=True)

    violations = predictor.check_independence()
    if violations:
        sig = sum(1 for v in violations if v["violation"] == "SIGNIFICANT")
        mod = sum(1 for v in violations if v["violation"] == "MODERATE")
        st.markdown(f"""
<div style="padding:12px 16px;background:rgba(217,119,6,0.08);border-radius:8px;
     border-left:3px solid {WARNING};margin-bottom:14px;font-size:0.88rem;">
  Found <b>{sig} significant</b> and <b>{mod} moderate</b> independence violations (p &lt; 0.01).
  This is expected — the model still works well in practice.
</div>
""", unsafe_allow_html=True)

        # Show sample contingency table for most-violated pair
        top_pair = violations[0]["pair"].split(" × ")
        if len(top_pair) == 2:
            ct = pd.crosstab(df[top_pair[0]], df[top_pair[1]],
                             rownames=[f"{top_pair[0]} (row)"],
                             colnames=[f"{top_pair[1]} (col)"])
            st.caption(f"Sample contingency table: {violations[0]['pair']}  (chi²={violations[0]['chi2']}, p={violations[0]['p_value']})")
            st.dataframe(ct, use_container_width=False)

        with st.expander(f"View all {len(violations)} violations"):
            st.dataframe(pd.DataFrame(violations), use_container_width=True, hide_index=True)
    else:
        st.info("No significant independence violations detected.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CALIBRATION ────────────────────────────────────────────────
    cal = predictor.check_calibration()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🎯 Calibration — Binomial Accuracy Check</div>', unsafe_allow_html=True)
    if cal and cal.get("status") == "calibrated":
        col = SUCCESS if cal["accuracy"] >= 70 else (WARNING if cal["accuracy"] >= 55 else DANGER)
        st.markdown(f"""
<div style="display:flex;gap:24px;align-items:center;">
  <div style="text-align:center;">
    <div style="font-size:2.2rem;font-weight:800;color:{col};">{cal['accuracy']}%</div>
    <div style="font-size:0.78rem;color:{MUTED};">Classification Accuracy</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:2.2rem;font-weight:800;">{cal['total_evaluated']}</div>
    <div style="font-size:0.78rem;color:{MUTED};">Verified Outcomes</div>
  </div>
  <div style="flex:1;font-size:0.88rem;color:{TEXT};">
    <b>{cal['interpretation']}</b><br>
    <span style="color:{MUTED};">Based on comparing risk category predictions vs. actual 18-month outcomes.</span>
  </div>
</div>
""", unsafe_allow_html=True)
    elif cal:
        st.info(cal.get("message", "Not enough verified outcomes yet."))
    else:
        st.info("No verified outcomes logged yet. Calibration will appear after 10+ outcomes are recorded.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE — DATABASE
# ══════════════════════════════════════════════════════════════════
elif page == "🗄️ Database":

    st.markdown("""
<div style="margin-bottom:28px;">
  <h1 style="font-size:2.1rem;margin-bottom:4px;">Record Ledger</h1>
  <div style="width:54px;height:3px;background:#B8860B;border-radius:2px;margin-bottom:14px;"></div>
  <p style="color:#475569;font-size:1rem;max-width:680px;margin:0;">
    Track all assessments and monitor data accumulation for model retraining.
  </p>
</div>
""", unsafe_allow_html=True)

    sd = get_data_stats()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Records", sd["mock_vendors"])
    c2.metric("Assessments Run",  sd["real_users"])
    c3.metric("Verified Outcomes",sd["with_feedback"])
    c4.metric("Total Records",    sd["total"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Volume chart
    user_path = "data/user_responses.csv"
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📆 Assessment Volume Over Time</div>', unsafe_allow_html=True)
    if os.path.exists(user_path):
        du = pd.read_csv(user_path)
        if len(du) > 0:
            du["timestamp"] = pd.to_datetime(du["timestamp"])
            du["date"]      = du["timestamp"].dt.date
            dy = du.groupby("date").size().reset_index(name="count")
            dy["cumulative"] = dy["count"].cumsum()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=dy["date"], y=dy["count"], name="Daily",
                marker_color=PRIMARY, opacity=0.8), secondary_y=False)
            fig.add_trace(go.Scatter(x=dy["date"], y=dy["cumulative"], name="Cumulative",
                line=dict(color=GOLD, width=3)), secondary_y=True)
            fig.update_xaxes(tickfont=PFONT)
            fig.update_yaxes(tickfont=PFONT, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(_layout(fig), use_container_width=True)
        else:
            st.info("No user assessments yet. Run an assessment to see data here.")
    else:
        st.info("No user assessments yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Recent records table
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">📄 Recent Assessments</div>', unsafe_allow_html=True)
    if os.path.exists(user_path):
        du = pd.read_csv(user_path)
        if len(du) > 0:
            show_cols = ["timestamp","vendor_id","total_score","survival_outcome"]
            st.dataframe(du[show_cols].tail(15), use_container_width=True, hide_index=True)
        else:
            st.info("No records yet.")
    else:
        st.info("No records yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Mock data preview
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">🏭 Baseline Mock Dataset Preview</div>', unsafe_allow_html=True)
    df_m = pd.read_csv("data/mock_data.csv")
    col_show = ["vendor_id","survival_outcome","total_score"] + [f"q{i}" for i in range(1,6)]
    st.dataframe(df_m[col_show].head(20), use_container_width=True, hide_index=True)
    st.caption(f"Showing first 20 of {len(df_m)} baseline records. Full data in data/mock_data.csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE — METHODOLOGY
# ══════════════════════════════════════════════════════════════════
elif page == "📚 Methodology":

    st.markdown("""
<div style="margin-bottom:28px;">
  <h1 style="font-size:2.1rem;margin-bottom:4px;">How SurvAI Works</h1>
  <div style="width:54px;height:3px;background:#B8860B;border-radius:2px;margin-bottom:14px;"></div>
  <p style="color:#475569;font-size:1rem;max-width:680px;margin:0;">
    A plain-English guide to every statistics concept used in this tool.
  </p>
</div>
""", unsafe_allow_html=True)

    concepts = [
        ("1. Prior Probability", "🔵", f"""
Before seeing any answers, we start with a guess: <b>P(Survived) = 30%</b>.
This is called the <b>prior</b> — the base rate from real-world research
(World Bank & IFC micro-business studies show ~70% of informal vendors
fail within 18 months). It is our starting belief before any evidence.
"""),
        ("2. Random Variable", "🎲", f"""
Each question answer (1, 2, or 3) is a <b>Discrete Random Variable</b> — it
takes one of a fixed set of values with certain probabilities.
For example, Q6 (savings buffer) has answer probabilities like:
P(answer=1) = 55% for failed vendors, P(answer=3) = 80% for survived vendors.
"""),
        ("3. Probability Distribution", "📊", f"""
For each question, we define two <b>Categorical Distributions</b> — one
for vendors who survived and one for those who failed.
A probability distribution tells us: for each possible answer (1, 2, 3),
how likely is it? All probabilities must sum to 1.0.
"""),
        ("4. Conditional Probability", "🔗", f"""
<b>P(answer | outcome)</b> = "Given we know this vendor survived (or failed),
how likely were they to give this specific answer?"
This is the core of our likelihood table.
For Q6: P(answer=3 | Survived) = 0.82 vs P(answer=3 | Failed) = 0.15.
Big difference = this question is very informative.
"""),
        ("5. Contingency Table", "📋", f"""
A <b>contingency table</b> is a grid that cross-counts two categorical
variables. For example, rows = Q1 answers (1/2/3), columns = Q6 answers.
Each cell counts how many vendors gave that combination.
The chi-squared test on this table tells us if the two questions are
correlated (which would violate our independence assumption).
"""),
        ("6. Multiplicative Rule", "✖️", f"""
Because Naive Bayes assumes the 15 questions are <b>independent</b>,
we can multiply their likelihoods:
<br><br>
P(Q1=3, Q6=3, … | Survived) = P(Q1=3|S) × P(Q6=3|S) × … × P(Q15=3|S)
<br><br>
This is the <b>Multiplicative Rule for independent events</b>.
We do this in log-space (adding logs) to avoid tiny floating-point numbers.
"""),
        ("7. Law of Total Probability", "⚖️", f"""
To find P(answer=k) for any question, we use the
<b>Law of Total Probability</b>:
<br><br>
P(answer=k) = P(answer=k | Survived) × P(Survived)
            + P(answer=k | Failed) × P(Failed)
<br><br>
This decomposes a probability over all possible outcomes.
It is used during Laplace smoothing to calibrate our likelihoods.
"""),
        ("8. Bayes' Theorem", "🧠", f"""
The engine of SurvAI. It combines the prior and evidence:
<br><br>
<b>P(Survived | answers) ∝ P(Survived) × ∏ P(answer_i | Survived)</b>
<br><br>
We compute the same for Failed, then normalise so the two posteriors sum to 1.
The ratio of numerators gives us the final survival probability.
This is Bayes' Theorem applied 15 times in a row.
"""),
        ("9. Normal Distribution", "🔔", f"""
Total resilience scores (sum of 15 answers, range 15–45) follow a
<b>Normal (Gaussian) Distribution</b> — the bell curve.
<br>• Survived: approximately N(μ=34, σ=4)
<br>• Failed: approximately N(μ=21, σ=5.5)
<br><br>
The bell curves overlap in the middle — vendors in that zone are hard
to classify. The Normal Distribution is used in the population chart
on the results page.
"""),
        ("10. Confidence Interval", "📏", f"""
A <b>95% Confidence Interval</b> means: if we ran this assessment on
many similar vendors, 95% of the resulting intervals would contain the
true survival probability.
<br><br>
We use the <b>Wilson Score</b> method (more accurate than simple ±1.96√(p/n)
near 0 and 1). It produces a range like "38% – 52%" around the prediction.
A narrower interval = more certainty.
"""),
        ("11. Binomial Distribution", "🎯", f"""
If N vendors each have probability p of surviving, the number who survive
follows <b>Binomial(N, p)</b>.
<br>• Mean = N × p &nbsp;|&nbsp; Std Dev = √(N × p × (1−p))
<br><br>
We also use Binomial logic for <b>calibration</b>: if our model predicts
70% accuracy, the number of correct predictions out of 50 follows Binomial(50, 0.70).
Shown interactively on the Analytics page.
"""),
        ("12. Poisson Distribution", "⚡", f"""
Q10 asks how many "zero-sale days" a vendor has per month.
Rare, random events in a fixed time window follow a <b>Poisson Distribution</b>:
<br><br>
P(X=k) = (λᵏ × e⁻λ) / k! &nbsp; where λ = average events per period
<br><br>
A vendor with λ=1 has a 37% chance of having zero bad days that month.
One with λ=5 has less than 1% chance. Shown interactively on Analytics.
"""),
        ("13. Central Tendency", "📍", f"""
<b>Central Tendency</b> measures where the "middle" of a data set is:
<br>• <b>Mean</b> — arithmetic average (sensitive to outliers)
<br>• <b>Median</b> — middle value when sorted (robust to outliers)
<br>• <b>Mode</b> — most frequently occurring value
<br><br>
In SurvAI: survived vendors score a mean of ~34/45, failed vendors ~21/45.
That 13-point gap is what makes the model work.
"""),
        ("14. Measure of Dispersion", "📐", f"""
<b>Dispersion</b> measures how spread out the data is:
<br>• <b>Std Dev / Variance</b> — average distance from the mean
<br>• <b>IQR</b> = Q3 − Q1 — spread of the middle 50%
<br>• <b>Range</b> = max − min
<br>• <b>CV</b> = Std Dev / Mean × 100 — relative spread
<br><br>
Higher dispersion within a group means vendors in that group are more
different from each other, making prediction harder.
"""),
        ("15. Probability Introduction", "🎓", f"""
<b>Probability</b> is a number between 0 and 1 that represents how
likely an event is. 0 = impossible, 1 = certain.
<br><br>
In SurvAI, every output is a probability:
the prior P(S)=0.30, the likelihoods P(answer|S), and the posterior
P(S|all answers). The whole tool is an exercise in moving from a
vague starting probability (30%) to a personalized one (e.g. 67%)
by applying evidence systematically.
"""),
    ]

    for title, icon, text in concepts:
        st.markdown(f"""
<div class="card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <span style="font-size:1.5rem;">{icon}</span>
    <h3 style="margin:0;font-size:1.1rem;color:{PRIMARY} !important;">{title}</h3>
  </div>
  <p style="color:{MUTED};font-size:0.91rem;line-height:1.75;margin:0;">{text}</p>
</div>
""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
<div class="card" style="border-left:4px solid {WARNING};background:rgba(217,119,6,0.04);">
  <div style="color:{WARNING};font-weight:700;margin-bottom:6px;">⚠️ Disclaimer</div>
  <p style="color:{MUTED};font-size:0.88rem;margin:0;line-height:1.7;">
    SurvAI is a statistics learning project. It is not professional financial advice,
    a credit score, or a bank recommendation. All data is stored locally and never shared.
    Predictions are based on mock data and should be treated as educational demonstrations.
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:4px 0;">
  <p style="color:#94A3B8;font-size:0.72rem;margin:0;">
    SurvAI v3.0 &nbsp;·&nbsp; Bayesian Inference Engine &nbsp;·&nbsp;
    15 Statistics Concepts &nbsp;·&nbsp; Local Storage Only
  </p>
</div>
""", unsafe_allow_html=True)