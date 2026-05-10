import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import json
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_PATH     = ROOT / "data/processed/09_with_audit_tiers.csv"
MANIFEST_PATH = ROOT / "models/xgboost/xgboost_manifest.json"

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_NAMES  = {1: "Standard Reminder", 2: "Soft Assist", 3: "Hardship Plan",
               4: "Genuine Distress",  5: "Strategic Default"}
TIER_COLORS = {1: "#10b981", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444", 5: "#9333ea"}
TIER_DIM    = "rgba(255,255,255,0.12)"

BRANCH_COLORS = {"UCI/Derived": "#6366f1", "BiLSTM": "#8b5cf6",
                 "FinBERT": "#06b6d4", "Sparkov": "#f59e0b"}

SHAP_DISPLAY = {
    "avg_pay_delay":       "Avg Payment Delay",
    "bilstm_anomaly_flag": "BiLSTM · Anomaly Flag",
    "bilstm_recon_error":  "BiLSTM · Recon Error",
    "inquiry_x_delay":     "Inquiry × Delay",
    "worst_month_delay":   "Worst Month Delay",
    "distress_avg":        "FinBERT · Avg Distress",
    "distress_max":        "FinBERT · Peak Distress",
    "mh_status_encoded":   "Mental Health Signal",
    "sp_total_spend":      "Sparkov · Total Spend",
    "months_delinquent":   "Months Delinquent",
}

NBA_MAP = {
    1: "Automated Reminder",   2: "Soft Assist SMS",
    3: "Offer Hardship Plan",  4: "RM Phone Call Required",
    5: "Legal Referral",
}

CHART_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(family="Space Mono, Courier New, monospace", size=11, color="#64748b"),
    margin        = dict(l=8, r=8, t=8, b=8),
    hoverlabel    = dict(bgcolor="#0d1425", font_color="#f1f5f9", font_size=12,
                         font_family="Space Mono, monospace", bordercolor="rgba(99,102,241,0.4)"),
    transition    = dict(duration=450, easing="cubic-in-out"),
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "SBDR · Intelligence Platform",
    page_icon             = "🛡️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&family=Orbitron:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

*, html, body, [class*="css"], .stApp {
    font-family: 'Space Mono', 'Courier New', monospace !important;
}
.material-icons, .material-icons-outlined, .material-icons-round {
    font-family: 'Material Icons' !important;
    font-feature-settings: 'liga' !important;
}
h1, h2, h3, .kpi-value, .mstrip-value, .section-label, .gradient-headline {
    font-family: 'Orbitron', 'Space Mono', monospace !important;
}

/* ── Base ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }
header [data-testid="stDecoration"],
header a[href*="streamlit.io"],
header .stDeployButton { display: none !important; }
.block-container { padding: 1.75rem 2.5rem 4rem 2.5rem !important; max-width: 100% !important; position: relative; z-index: 1; }

.stApp {
    background-color: #060b18 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 8% 40%,  rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 55% 40% at 88% 14%, rgba(139,92,246,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 95%, rgba(6,182,212,0.05)  0%, transparent 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e20 0%, #060b18 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCheckbox span { color: #475569 !important; font-size: 13px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: rgba(15,23,42,0.8) !important;
    border-color: rgba(99,102,241,0.25) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.05) !important; margin: 1rem 0; }
[data-testid="stSidebar"] button {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] button:hover {
    background: rgba(99,102,241,0.25) !important;
}

/* ── Sidebar collapse / expand toggle ── */
[data-testid="collapsedControl"] {
    background: rgba(99,102,241,0.2) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 0 8px 8px 0 !important;
    color: #a5b4fc !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(99,102,241,0.35) !important;
}
[data-testid="collapsedControl"] svg {
    fill: #a5b4fc !important;
    stroke: #a5b4fc !important;
}
/* Fix: global Space Mono override breaks Material Symbols ligatures.
   Zero out the broken text and replace with CSS arrow characters. */
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] p {
    font-size: 0 !important;
    line-height: 0 !important;
}
[data-testid="collapsedControl"] span::after,
[data-testid="collapsedControl"] p::after {
    content: '❯';
    font-size: 16px !important;
    color: #a5b4fc;
    font-family: 'Space Mono', monospace !important;
    line-height: 1 !important;
}
button[data-testid="baseButton-secondary"][aria-label="Close sidebar"],
button[data-testid="baseButton-secondary"][aria-label="Open sidebar"] {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
button[data-testid="baseButton-secondary"][aria-label="Close sidebar"] span,
button[data-testid="baseButton-secondary"][aria-label="Open sidebar"] span {
    font-size: 0 !important;
    line-height: 0 !important;
}
button[data-testid="baseButton-secondary"][aria-label="Close sidebar"] span::after {
    content: '❮';
    font-size: 16px !important;
    color: #a5b4fc;
    font-family: 'Space Mono', monospace !important;
    line-height: 1 !important;
}
button[data-testid="baseButton-secondary"][aria-label="Open sidebar"] span::after {
    content: '❯';
    font-size: 16px !important;
    color: #a5b4fc;
    font-family: 'Space Mono', monospace !important;
    line-height: 1 !important;
}
section[data-testid="stSidebar"] > div > div > button {
    color: #a5b4fc !important;
}

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 8px  rgba(147,51,234,0.4), 0 4px 24px rgba(0,0,0,0.5); }
    50%      { box-shadow: 0 0 20px rgba(147,51,234,0.7), 0 4px 24px rgba(0,0,0,0.5); }
}
@keyframes borderShimmer {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}
@keyframes scanline {
    from { transform: translateX(-100%); }
    to   { transform: translateX(100%);  }
}

/* ── Cards ── */
.card {
    background: rgba(10, 15, 30, 0.85);
    border-radius: 14px;
    padding: 22px 24px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.55s cubic-bezier(0.22,0.61,0.36,1) both;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.35) 40%, rgba(139,92,246,0.35) 60%, transparent 100%);
}
.card:hover {
    border-color: rgba(99,102,241,0.22);
    box-shadow: 0 8px 40px rgba(0,0,0,0.55), 0 0 0 1px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    transform: translateY(-1px);
}
.card-title { font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.card-sub   { font-size: 11.5px; color: #3d4f6b; font-weight: 400; margin-bottom: 14px; }

/* ── Section label ── */
.section-label {
    font-size: 10px; font-weight: 800; color: #6366f1;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin: 32px 0 16px 0;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4) 0%, transparent 100%);
}

/* ── Model health strip ── */
.mstrip-card {
    background: rgba(10,15,30,0.85);
    border-radius: 14px; padding: 18px 20px;
    border: 1px solid rgba(255,255,255,0.07); text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    backdrop-filter: blur(20px);
    position: relative; overflow: hidden;
    animation: slideUp 0.5s cubic-bezier(0.22,0.61,0.36,1) 0.25s both;
    transition: border-color 0.25s ease, transform 0.2s ease;
}
.mstrip-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129,140,248,0.4), transparent);
}
.mstrip-card:hover { border-color: rgba(129,140,248,0.25); transform: translateY(-1px); }
.mstrip-label { font-size: 10px; font-weight: 700; color: #3d4f6b; text-transform: uppercase; letter-spacing: 0.1em; }
.mstrip-value { font-size: 22px; font-weight: 900; color: #818cf8; margin-top: 5px; letter-spacing: 1px; font-family: 'Orbitron', 'Space Mono', monospace !important; }
.mstrip-sub   { font-size: 11px; color: #64748b; margin-top: 3px; }

/* ── Audit panel rows ── */
.audit-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.audit-key { font-size: 12px; color: #3d4f6b; font-weight: 500; }
.audit-val { font-size: 12px; color: #cbd5e1; font-weight: 600; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; white-space: nowrap; }

/* ── Filter pills ── */
.filter-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; align-items: center; }
.pill { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.pill-blue   { background: rgba(37,99,235,0.15);  color: #60a5fa; border: 1px solid rgba(37,99,235,0.3);  }
.pill-indigo { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
.pill-amber  { background: rgba(217,119,6,0.15);  color: #fbbf24; border: 1px solid rgba(217,119,6,0.3);  }

/* ── Dataframe dark ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stDataFrame"] th { background: rgba(8,12,26,0.95) !important; color: #3d4f6b !important; border-bottom: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; background: rgba(6,10,20,0.8) !important; }
[data-testid="stDataFrame"] tr:hover td { background: rgba(99,102,241,0.06) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

/* ── Input / slider dark ── */
input[type="range"] { accent-color: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str, border: str, sub_color: str = "#475569") -> None:
    num_m    = re.search(r"[\d,]+\.?\d*", value)
    raw_num  = float(num_m.group().replace(",", "")) if num_m else 0
    prefix   = value[:num_m.start()] if num_m else ""
    suffix   = value[num_m.end():]   if num_m else ""
    decimals = len(value.split(".")[-1]) if ("." in value and num_m) else 0
    components.html(f"""
<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{background:transparent;overflow:hidden;font-family:'Space Mono','Courier New',monospace}}
.c{{
  background:rgba(10,15,30,0.92);border-radius:14px;padding:20px 22px 18px;
  border:1px solid rgba(255,255,255,0.07);border-left:3px solid {border};
  position:relative;overflow:hidden;
  animation:su .55s cubic-bezier(.22,.61,.36,1) both;
  transition:transform .2s ease,box-shadow .2s ease;
  box-shadow:0 4px 28px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.04);
}}
.c::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent)}}
.c::after{{content:'';position:absolute;bottom:-20px;right:-20px;width:90px;height:90px;
  background:radial-gradient(circle,{border}20 0%,transparent 70%);pointer-events:none}}
@keyframes su{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
.lbl{{font-size:9px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:.15em;margin-bottom:10px;font-family:'Space Mono',monospace}}
.val{{font-size:24px;font-weight:900;color:#f1f5f9;line-height:1.1;margin-bottom:7px;
  letter-spacing:-.5px;font-variant-numeric:tabular-nums;font-family:'Orbitron','Space Mono',monospace}}
.sub{{font-size:11.5px;font-weight:500;color:{sub_color}}}
</style></head><body>
<div class="c">
  <div class="lbl">{label}</div>
  <div class="val" id="v">{value}</div>
  <div class="sub">{sub}</div>
</div>
<script>
(function(){{
  var el=document.getElementById('v');
  var end={raw_num},dec={decimals},pre='{prefix}',suf='{suffix}';
  var dur=1500,t0=performance.now();
  function ease(t){{return 1-Math.pow(1-t,3);}}
  function tick(now){{
    var p=Math.min((now-t0)/dur,1),v=end*ease(p);
    var s=dec>0?v.toFixed(dec):Math.round(v).toLocaleString();
    el.textContent=pre+s+suf;
    if(p<1)requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script></body></html>""", height=108, scrolling=False)


def card_header(title: str, sub: str = "") -> None:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="card-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def badge(text: str, bg: str, color: str) -> str:
    return f'<span class="badge" style="background:{bg};color:{color};">{text}</span>'


def audit_row(key: str, val_html: str) -> str:
    return (f'<div class="audit-row">'
            f'<span class="audit-key">{key}</span>'
            f'<span class="audit-val">{val_html}</span>'
            f'</div>')


def get_xai_signal(row: pd.Series) -> str:
    if bool(row["bilstm_anomaly_flag"]):      return "BiLSTM · Payment Anomaly"
    if row["distress_avg"] > 0.65:            return "FinBERT · High Distress"
    if row["avg_pay_delay"] > 2:              return "UCI · Severe Delinquency"
    return "XGBoost · Multi-Signal"


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)

df       = load_data()
manifest = load_manifest()

# Pre-compute spotlight case (Tier 5, Rule C preferred) — reused across sections
_t5_df         = df[df["recovery_tier_final"] == 5]
_rule_c_cases  = _t5_df[_t5_df["audit_rule_c"] == True]
spotlight_case = _rule_c_cases.iloc[0] if len(_rule_c_cases) > 0 else _t5_df.iloc[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:4px;">
      <div style="font-size:16px;font-weight:800;letter-spacing:-0.3px;
                  background:linear-gradient(135deg,#e2e8f0,#a5b4fc);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        🛡️ SBDR Platform
      </div>
      <p style="color:#64748b;font-size:11px;margin:3px 0 0 0;">Intelligence · Collections · Compassion</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="color:#3d4f6b;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Portfolio Filters</p>', unsafe_allow_html=True)
    selected_tiers = st.multiselect(
        "Recovery Tier", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5],
        format_func=lambda x: f"T{x} · {TIER_NAMES[x]}",
    )
    min_distress = st.slider("Min Distress Score", 0.0, 1.0, 0.0, 0.01)
    max_distress = st.slider("Max Distress Score", 0.0, 1.0, 1.0, 0.01)
    anomaly_only = st.checkbox("Anomaly Flagged Only", value=False)
    st.markdown("---")

    st.markdown('<p style="color:#3d4f6b;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Model Performance</p>', unsafe_allow_html=True)
    m = manifest["metrics"]
    for lbl, val in [("Accuracy",    f"{m['accuracy']*100:.1f}%"),
                     ("AUC-ROC",     f"{m['auc_roc_macro']:.3f}"),
                     ("F1 Weighted", f"{m['f1_weighted']*100:.1f}%"),
                     ("F1 Macro",    f"{m['f1_macro']*100:.1f}%")]:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:12px;color:#64748b;">{lbl}</span>'
            f'<span style="font-size:12px;font-weight:700;color:#818cf8;">{val}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")

    st.markdown('<p style="color:#3d4f6b;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:12px;">30,000 customers · 142 features<br>UCI · LC · Sparkov · FinBERT · BiLSTM</p>', unsafe_allow_html=True)

    if st.button("↺ Reset Filters", use_container_width=True):
        st.rerun()


# ── Filtered dataset ──────────────────────────────────────────────────────────
active_tiers = selected_tiers if selected_tiers else [1, 2, 3, 4, 5]
mask = (
    df["recovery_tier_final"].isin(active_tiers)
    & (df["distress_avg"] >= min_distress)
    & (df["distress_avg"] <= max_distress)
)
if anomaly_only:
    mask &= df["bilstm_anomaly_flag"] == True
filtered_df = df[mask]

filters_active = (
    set(active_tiers) != {1, 2, 3, 4, 5}
    or min_distress > 0
    or max_distress < 1.0
    or anomaly_only
)

# ── Global KPIs (always total portfolio) ─────────────────────────────────────
total_customers    = len(df)
total_credit_limit = df["LIMIT_BAL"].sum()
avg_distress_all   = df["distress_avg"].mean()
tier4_count        = int((df["recovery_tier_final"] == 4).sum())
tier5_count        = int((df["recovery_tier_final"] == 5).sum())
high_risk_limit    = df[df["recovery_tier_final"].isin([4, 5])]["LIMIT_BAL"].sum()
anomaly_count      = int(df["bilstm_anomaly_flag"].sum())

distress_color  = "#f59e0b" if avg_distress_all > 0.50 else "#10b981"
distress_status = "⬆ Elevated · Above 0.50" if avg_distress_all > 0.50 else "● Stable"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:20px;">
  <div style="display:inline-flex;align-items:center;gap:7px;
              background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
              color:#818cf8;padding:4px 14px;border-radius:20px;
              font-size:10.5px;font-weight:700;letter-spacing:.08em;
              text-transform:uppercase;margin-bottom:10px;">
    <span style="width:6px;height:6px;border-radius:50%;background:#6366f1;
                 box-shadow:0 0 8px rgba(99,102,241,0.8);display:inline-block;
                 animation:none;"></span>
    DATA 606 Capstone &nbsp;·&nbsp; UMBC &nbsp;·&nbsp; Live Portfolio Intelligence
  </div>
  <h1 style="margin:0;font-size:26px;font-weight:900;letter-spacing:1px;line-height:1.2;
             font-family:'Orbitron','Space Mono',monospace;
             background:linear-gradient(135deg,#e2e8f0 0%,#a5b4fc 45%,#c084fc 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;text-transform:uppercase;">
    Sentimental-Behavioral Debt Recovery
  </h1>
  <p style="color:#64748b;font-size:13px;margin:6px 0 0 0;font-weight:400;">
    Multi-Modal AI Framework &nbsp;·&nbsp; FinBERT · BiLSTM · XGBoost · Audit Layer &nbsp;·&nbsp; 30,000 Customers · 142 Features
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    kpi_card("Monitored Accounts", f"{total_customers:,}",
             f"Credit Limit Pool: ${total_credit_limit/1e9:.2f}B", "#6366f1")
with k2:
    kpi_card("High-Risk Accounts", f"{tier4_count + tier5_count:,}",
             f"Tier 4 + 5 · ${high_risk_limit/1e9:.2f}B Exposure", "#ef4444", "#ef4444")
with k3:
    kpi_card("Avg Portfolio Distress", f"{avg_distress_all:.2f}",
             distress_status, "#f59e0b", distress_color)
with k4:
    kpi_card("BiLSTM Anomalies", f"{anomaly_count:,}",
             "5.0% of portfolio · Recon Error > P95", "#8b5cf6", "#8b5cf6")

st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

# ── Model performance strip ───────────────────────────────────────────────────
section_label("Model Health")
m1, m2, m3, m4 = st.columns(4, gap="medium")
for col, (lbl, val, sub) in zip(
    [m1, m2, m3, m4],
    [("Accuracy",    f"{m['accuracy']*100:.1f}%",      "Overall Classification"),
     ("AUC-ROC",     f"{m['auc_roc_macro']:.3f}",      "Macro One-vs-Rest"),
     ("F1 Weighted", f"{m['f1_weighted']*100:.1f}%",   "Weighted by Class Size"),
     ("F1 Macro",    f"{m['f1_macro']*100:.1f}%",      "Unweighted All Tiers")],
):
    with col:
        st.markdown(
            f'<div class="mstrip-card"><div class="mstrip-label">{lbl}</div>'
            f'<div class="mstrip-value">{val}</div><div class="mstrip-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

# ── Row 1: Tier distribution + SHAP ──────────────────────────────────────────
section_label("Portfolio Analysis")
c1, c2 = st.columns([1.55, 1], gap="medium")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header(
        "Recovery Tier Distribution",
        f"{'Filtered: ' + str(len(active_tiers)) + ' tiers selected · ' if filters_active else ''}"
        "Final classification after B3.5 audit layer",
    )
    tier_counts   = df["recovery_tier_final"].value_counts().sort_index()
    pct           = (tier_counts / len(df) * 100).round(1)
    bar_colors    = [TIER_COLORS[i] if i in active_tiers else TIER_DIM for i in tier_counts.index]
    text_labels   = [f"{c:,}  ({p}%)" for c, p in zip(tier_counts.values, pct.values)]

    fig_tier = go.Figure(go.Bar(
        x=[f"T{i} · {TIER_NAMES[i]}" for i in tier_counts.index],
        y=tier_counts.values.tolist(),
        marker_color=bar_colors, marker_line_width=0,
        text=text_labels, textposition="outside",
        textfont=dict(size=11, family="Space Mono, monospace", color="#94a3b8"),
        hovertemplate="<b>%{x}</b><br>%{y:,} customers<extra></extra>",
    ))
    fig_tier.update_layout(
        **CHART_BASE, height=290,
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False,
                   title=dict(text="Customers", font=dict(size=11))),
    )
    fig_tier.update_yaxes(range=[0, tier_counts.max() * 1.22])
    st.plotly_chart(fig_tier, use_container_width=True, config={"displayModeBar": False}, key="tier_chart")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Top SHAP Feature Importances", "Mean |SHAP| value — colored by AI branch")

    top_feats  = manifest["top_features_by_shap"][:8]
    names      = [SHAP_DISPLAY.get(f["feature"], f["feature"]) for f in top_feats][::-1]
    vals       = [f["mean_abs_shap"] for f in top_feats][::-1]
    bar_clrs   = [BRANCH_COLORS.get(f["branch"], "#94a3b8") for f in top_feats][::-1]

    legend_html = " &nbsp; ".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#94a3b8;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{c};box-shadow:0 0 5px {c}80;display:inline-block;"></span>{b}</span>'
        for b, c in BRANCH_COLORS.items()
    )
    st.markdown(f'<div style="margin-bottom:8px;">{legend_html}</div>', unsafe_allow_html=True)

    fig_shap = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=bar_clrs, marker_line_width=0,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        textfont=dict(size=10.5, family="Space Mono, monospace", color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        **CHART_BASE, height=278,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False,
                   title=dict(text="Mean |SHAP|", font=dict(size=11))),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11)),
    )
    fig_shap.update_xaxes(range=[0, max(vals) * 1.22])
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False}, key="shap_chart")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: Distress histogram + Branch donut ──────────────────────────────────
c3, c4 = st.columns([1.55, 1], gap="medium")

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    filt_label = (f"{len(filtered_df):,} customers matching filters"
                  if filters_active else "All 30,000 customers")
    card_header("Distress Score Distribution", f"{filt_label} · per-tier overlay")

    fig_hist = go.Figure()
    plot_tiers = active_tiers if active_tiers else [1, 2, 3, 4, 5]
    for tier in sorted(plot_tiers):
        tier_data = filtered_df[filtered_df["recovery_tier_final"] == tier]["distress_avg"]
        if len(tier_data) == 0:
            continue
        fig_hist.add_trace(go.Histogram(
            x=tier_data, name=f"T{tier} · {TIER_NAMES[tier]}",
            marker_color=TIER_COLORS[tier], marker_line_width=0,
            opacity=0.72, nbinsx=40,
            hovertemplate=f"<b>T{tier}: {TIER_NAMES[tier]}</b><br>Distress: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig_hist.update_layout(
        **CHART_BASE, height=285, barmode="overlay",
        xaxis=dict(showgrid=False, showline=False, zeroline=False,
                   title=dict(text="Distress Score", font=dict(size=11)),
                   range=[0, 1], tickformat=".1f"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False,
                   title=dict(text="Customers", font=dict(size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10.5, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False}, key="dist_hist")
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("AI Branch Contributions", "Mean |SHAP| share — which branch drives predictions")

    branch_contrib = manifest["branch_shap_contribution"]
    total_shap     = sum(branch_contrib.values())
    b_labels       = list(branch_contrib.keys())
    b_vals         = [v / total_shap * 100 for v in branch_contrib.values()]
    b_colors       = [BRANCH_COLORS.get(b, "#94a3b8") for b in b_labels]

    fig_donut = go.Figure(go.Pie(
        labels=b_labels, values=b_vals, hole=0.62,
        marker=dict(colors=b_colors, line=dict(color="rgba(6,11,24,0.7)", width=2)),
        textinfo="label+percent", textfont=dict(size=11, family="Space Mono, monospace"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}% of SHAP<extra></extra>",
        pull=[0.04 if b == "UCI/Derived" else 0 for b in b_labels],
    ))
    fig_donut.update_layout(
        **CHART_BASE, height=285,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{max(b_vals):.0f}%</b><br><span style='font-size:10px'>UCI</span>",
            x=0.5, y=0.5, font=dict(size=14, color="#e2e8f0"), showarrow=False,
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False}, key="donut_chart")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Row 2.5: Multi-Branch Customer Evidence ───────────────────────────────────
section_label(f"Multi-Branch Customer Evidence · Case: {spotlight_case['customer_id']}  (Tier 5 · Strategic Default)")
ev1, ev2, ev3 = st.columns(3, gap="medium")

with ev1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Branch 1 · FinBERT Sentiment Progression",
                "Per-turn distress scores extracted from customer chat log")

    turn_scores = [
        spotlight_case["distress_turn_1"],
        spotlight_case["distress_turn_2"],
        spotlight_case["distress_turn_3"],
    ]
    turn_labels = ["Turn 1", "Turn 2", "Turn 3"]
    turn_colors = ["#06b6d4" if s < 0.5 else "#f59e0b" if s < 0.75 else "#ef4444"
                   for s in turn_scores]

    fig_turns = go.Figure(go.Bar(
        x=turn_labels, y=turn_scores,
        marker_color=turn_colors, marker_line_width=0,
        text=[f"{s:.3f}" for s in turn_scores], textposition="outside",
        textfont=dict(size=11, family="Space Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Distress: %{y:.3f}<extra></extra>",
    ))
    fig_turns.add_hline(
        y=spotlight_case["distress_avg"], line_dash="dot", line_color="#94a3b8",
        annotation_text=f"avg {spotlight_case['distress_avg']:.3f}",
        annotation_font_size=10, annotation_font_color="#64748b",
    )
    fig_turns.update_layout(
        **CHART_BASE, height=195,
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False,
                   zeroline=False, range=[0, 1.15],
                   title=dict(text="Distress Score", font=dict(size=11))),
    )
    st.plotly_chart(fig_turns, use_container_width=True,
                    config={"displayModeBar": False}, key="ev_turns_chart")

    shift_val = spotlight_case["distress_shift"]
    shift_color = "#16a34a" if shift_val < 0 else "#dc2626"
    shift_label = f"{'▼' if shift_val < 0 else '▲'} {abs(shift_val):.3f} ({'de-escalating' if shift_val < 0 else 'escalating'})"
    st.markdown(
        f'<div style="margin-bottom:10px;padding:6px 10px;background:rgba(255,255,255,0.03);border-radius:6px;">'
        f'<span style="font-size:10.5px;font-weight:600;color:#3d4f6b;">Sentiment Shift (T3→T1): </span>'
        f'<span style="font-size:11px;font-weight:700;color:{shift_color};">{shift_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i, (col_name, score) in enumerate(
        zip(["chat_turn_1", "chat_turn_2", "chat_turn_3"], turn_scores), 1
    ):
        text = spotlight_case[col_name]
        text = "(no chat recorded)" if pd.isna(text) else str(text)
        border = turn_colors[i - 1]
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:8px 12px;'
            f'margin-bottom:6px;border-left:3px solid {border};">'
            f'<p style="font-size:10px;font-weight:700;color:#3d4f6b;margin:0 0 3px 0;letter-spacing:.08em;text-transform:uppercase;">'
            f'Turn {i} · distress {score:.3f}</p>'
            f'<p style="font-size:11.5px;color:#94a3b8;margin:0;line-height:1.6;">'
            f'{text[:200]}{"…" if len(text) > 200 else ""}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with ev2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Branch 2 · BiLSTM Payment Sequence",
                "6-month payment status (M1 = most recent) · green=on-time, red=delayed")

    pay_cols   = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    pay_vals   = [float(spotlight_case[c]) for c in pay_cols]
    pay_months = ["M1", "M2", "M3", "M4", "M5", "M6"]
    pay_colors = ["#10b981" if v <= 0 else "#f59e0b" if v <= 2 else "#ef4444"
                  for v in pay_vals]

    fig_pay = go.Figure(go.Bar(
        x=pay_months, y=pay_vals,
        marker_color=pay_colors, marker_line_width=0,
        text=[str(int(v)) for v in pay_vals], textposition="outside",
        textfont=dict(size=11, family="Space Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Delay status: %{y}<extra></extra>",
    ))
    fig_pay.add_hline(y=0, line_color="#e2e8f0", line_width=1)
    fig_pay.update_layout(
        **CHART_BASE, height=195,
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False,
                   title=dict(text="Pay Status (months delayed)", font=dict(size=11))),
    )
    st.plotly_chart(fig_pay, use_container_width=True,
                    config={"displayModeBar": False}, key="ev_pay_chart")

    recon_err = float(spotlight_case["bilstm_recon_error"])
    anomaly   = bool(spotlight_case["bilstm_anomaly_flag"])
    lc_inc    = spotlight_case.get("lc_annual_inc_mean")
    lc_dti    = spotlight_case.get("lc_dti_mean")

    bilstm_stats = [
        ("Recon Error",     badge(f"{recon_err:.5f}",
                                  "#fef2f2" if anomaly else "#f0fdf4",
                                  "#dc2626" if anomaly else "#16a34a")),
        ("Anomaly Flag",    badge("Detected ⚠", "rgba(239,68,68,0.15)", "#f87171") if anomaly
                            else badge("Clear ✓", "rgba(16,185,129,0.15)", "#34d399")),
        ("Avg Pay Delay",   badge(f"{spotlight_case['avg_pay_delay']:.2f} mo", "rgba(245,158,11,0.15)", "#fbbf24")),
        ("Worst Month",     badge(f"{int(spotlight_case['worst_month_delay'])} mo delay", "rgba(245,158,11,0.15)", "#fbbf24")),
    ]
    if lc_inc is not None and not pd.isna(lc_inc):
        bilstm_stats.append(("Est. Annual Income", badge(f"${float(lc_inc):,.0f}", "rgba(16,185,129,0.15)", "#34d399")))
    if lc_dti is not None and not pd.isna(lc_dti):
        bilstm_stats.append(("Debt-to-Income Ratio", badge(f"{float(lc_dti):.1f}%", "rgba(59,130,246,0.15)", "#60a5fa")))

    rows_html = "".join(audit_row(k, v) for k, v in bilstm_stats)
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px 12px;border:1px solid rgba(255,255,255,0.05);margin-top:8px;">'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with ev3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Branch 3 · XGBoost Tier Confidence",
                "Softmax probability per recovery tier · all 5 tiers shown")

    prob_labels = [f"T{i} · {TIER_NAMES[i]}" for i in range(1, 6)]
    prob_vals   = [float(spotlight_case[f"tier_prob_{i}"]) for i in range(1, 6)]
    prob_colors = [TIER_COLORS[i] for i in range(1, 6)]

    fig_prob = go.Figure(go.Bar(
        x=prob_labels, y=prob_vals,
        marker_color=prob_colors, marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in prob_vals], textposition="outside",
        textfont=dict(size=11, family="Space Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Probability: %{y:.4f}<extra></extra>",
    ))
    fig_prob.update_layout(
        **CHART_BASE, height=195,
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=10.5)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False,
                   zeroline=False, tickformat=".2f",
                   title=dict(text="Probability", font=dict(size=11))),
    )
    fig_prob.update_yaxes(range=[0, max(prob_vals) * 1.25])
    st.plotly_chart(fig_prob, use_container_width=True,
                    config={"displayModeBar": False}, key="ev_prob_chart")

    sparkov_stats = [
        ("Total Spend",      badge(f"${float(spotlight_case['sp_total_spend']):,.2f}", "rgba(59,130,246,0.15)", "#60a5fa")),
        ("Monthly Avg",      badge(f"${float(spotlight_case['sp_avg_monthly_spend']):,.2f}", "rgba(59,130,246,0.15)", "#60a5fa")),
        ("Spend Volatility", badge(f"{float(spotlight_case['sp_spend_volatility']):,.2f}", "rgba(245,158,11,0.15)", "#fbbf24")),
        ("Fraud Rate",       badge(f"{float(spotlight_case['sp_fraud_rate']):.3f}", "rgba(239,68,68,0.15)", "#f87171")),
        ("Num Transactions", badge(f"{int(spotlight_case['sp_num_transactions']):,}", "rgba(16,185,129,0.15)", "#34d399")),
    ]
    rows_html = "".join(audit_row(k, v) for k, v in sparkov_stats)
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px 12px;border:1px solid rgba(255,255,255,0.05);margin-top:8px;">'
        f'<p style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;'
        f'letter-spacing:0.07em;margin:0 0 6px 0;">Sparkov Transaction Signals</p>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

# ── Row 3: Roster + Audit ─────────────────────────────────────────────────────
section_label("Operational Execution")

# Filter pills
if filters_active:
    pills = []
    if set(active_tiers) != {1,2,3,4,5}:
        for t in active_tiers:
            pills.append(f'<span class="pill pill-blue">T{t}: {TIER_NAMES[t]}</span>')
    if min_distress > 0 or max_distress < 1.0:
        pills.append(f'<span class="pill pill-amber">Distress {min_distress:.2f}–{max_distress:.2f}</span>')
    if anomaly_only:
        pills.append('<span class="pill pill-indigo">Anomaly Only</span>')
    if pills:
        st.markdown(
            f'<div class="filter-pills"><span style="font-size:11px;color:#3d4f6b;font-weight:700;letter-spacing:.04em;">Active filters:</span>'
            + "".join(pills) + '</div>',
            unsafe_allow_html=True,
        )

roster_col, audit_col = st.columns([2, 1.4], gap="medium")

with roster_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Inline search
    search_col, count_col = st.columns([3, 1])
    with search_col:
        search = st.text_input(
            "", placeholder="Search by Customer ID (e.g. SBDR_0042...)",
            label_visibility="collapsed", key="customer_search",
        )
    with count_col:
        st.markdown(
            f'<div style="padding-top:8px;text-align:right;font-size:12px;color:#3d4f6b;font-weight:500;">'
            f'{len(filtered_df):,} matching</div>',
            unsafe_allow_html=True,
        )

    # Apply search
    display_df = filtered_df.copy()
    if search:
        display_df = display_df[display_df["customer_id"].str.contains(search.upper(), na=False)]

    card_header(
        "Customer Roster",
        "Sorted by distress score · Next Best Action assigned per tier",
    )

    roster_raw = display_df.nlargest(12, "distress_avg")
    df_roster  = pd.DataFrame({
        "Customer":          roster_raw["customer_id"].values,
        "Tier":              [f"T{t}" for t in roster_raw["recovery_tier_final"]],
        "Distress":          roster_raw["distress_avg"].round(3).values,
        "Anomaly":           ["Yes" if x else "—" for x in roster_raw["bilstm_anomaly_flag"]],
        "Avg Delay (mo)":    roster_raw["avg_pay_delay"].round(2).values,
        "XAI Signal":        [get_xai_signal(r) for _, r in roster_raw.iterrows()],
        "Next Best Action":  [NBA_MAP[t] for t in roster_raw["recovery_tier_final"]],
    })
    st.dataframe(
        df_roster, use_container_width=True, hide_index=True,
        column_config={
            "Customer":       st.column_config.TextColumn("Customer", width="medium"),
            "Tier":           st.column_config.TextColumn("Tier", width="small"),
            "Distress":       st.column_config.ProgressColumn(
                                  "Distress", min_value=0, max_value=1, format="%.3f", width="medium"),
            "Anomaly":        st.column_config.TextColumn("Anomaly", width="small"),
            "Avg Delay (mo)": st.column_config.NumberColumn("Avg Delay (mo)", format="%.2f", width="small"),
            "XAI Signal":     st.column_config.TextColumn("XAI Signal", width="large"),
            "Next Best Action": st.column_config.TextColumn("Next Best Action", width="large"),
        },
        height=370,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with audit_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("B3.5 Audit Layer · Case Review",
                "Rule-based safety net for Tier 5 confirmation")

    t5_df     = _t5_df
    escalated = int(df["audit_escalated"].sum())
    rows_html = "".join([
        audit_row("Strategic Default (T5)",  badge(f"{len(t5_df):,} customers", "rgba(239,68,68,0.15)", "#f87171")),
        audit_row("Audit Escalations",       badge(f"+{escalated} escalated", "rgba(239,68,68,0.15)", "#f87171")),
        audit_row("Rule A · Label Override", badge(f"{int(df['audit_rule_a'].sum()):,}", "rgba(59,130,246,0.15)", "#60a5fa")),
        audit_row("Rule B · Tier5 Prob≥0.15",badge(f"{int(df['audit_rule_b'].sum()):,}", "rgba(59,130,246,0.15)", "#60a5fa")),
        audit_row("Rule C · Fraud+Anomaly",  badge(f"{int(df['audit_rule_c'].sum()):,}", "rgba(139,92,246,0.15)", "#c084fc")),
    ])
    st.markdown(f'<div style="margin-bottom:18px;">{rows_html}</div>', unsafe_allow_html=True)

    case = spotlight_case

    rules_hit = []
    if bool(case["audit_rule_a"]): rules_hit.append("A")
    if bool(case["audit_rule_b"]): rules_hit.append("B")
    if bool(case["audit_rule_c"]): rules_hit.append("C")
    if bool(case["audit_rule_d"]): rules_hit.append("D")
    rule_badge = badge("Rule " + "+".join(rules_hit) if rules_hit else "Direct", "rgba(139,92,246,0.15)", "#c084fc")

    case_rows = "".join([
        audit_row("Case ID",        f'<code style="font-size:11px;">{case["customer_id"]}</code>'),
        audit_row("Distress Score", badge(f'{case["distress_avg"]:.3f}', "rgba(239,68,68,0.15)", "#f87171")),
        audit_row("BiLSTM Anomaly", badge("Detected", "rgba(239,68,68,0.15)", "#f87171") if bool(case["bilstm_anomaly_flag"]) else badge("Clear", "rgba(16,185,129,0.15)", "#34d399")),
        audit_row("Payment Delay",  badge(f'{case["avg_pay_delay"]:.2f} mo', "rgba(245,158,11,0.15)", "#fbbf24")),
        audit_row("Fraud Rate",     badge(f'{case["sp_fraud_rate"]:.3f}', "rgba(239,68,68,0.15)", "#f87171")),
        audit_row("Default Label",  badge("Defaulted", "rgba(239,68,68,0.15)", "#f87171") if int(case["default payment next month"]) == 1 else badge("No Default", "rgba(16,185,129,0.15)", "#34d399")),
        audit_row("Audit Rule",     rule_badge),
    ])
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:14px 16px;border:1px solid rgba(255,255,255,0.05);">'
        f'<p style="font-size:10px;font-weight:700;color:#3d4f6b;text-transform:uppercase;letter-spacing:0.12em;margin:0 0 8px 0;">Case Spotlight · {case["customer_id"]}</p>'
        f'{case_rows}'
        f'<div style="margin-top:12px;padding:10px 12px;background:rgba(239,68,68,0.1);border-radius:8px;border-left:3px solid #ef4444;">'
        f'<p style="font-size:12px;font-weight:700;color:#ef4444;margin:0;letter-spacing:0.02em;">Tier 5 — Strategic Default Confirmed</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Fairness & Demographic Audit (C2) ─────────────────────────────────────────
section_label("Fairness & Demographic Audit (C2) — Tier Distribution Across Demographics")

def _tier_pct_pivot(group_col: str, label_map: dict | None = None) -> pd.DataFrame:
    tmp = df[["recovery_tier_final", group_col]].copy()
    if label_map:
        tmp[group_col] = tmp[group_col].map(label_map).fillna("Other")
    pivot = tmp.groupby([group_col, "recovery_tier_final"]).size().unstack(fill_value=0)
    return pivot.div(pivot.sum(axis=1), axis=0) * 100

def _fairness_bar(pivot: pd.DataFrame, key: str) -> go.Figure:
    fig = go.Figure()
    for tier in sorted(df["recovery_tier_final"].unique()):
        if tier not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            name=f"T{tier}",
            x=[str(g) for g in pivot.index.tolist()],
            y=pivot[tier].tolist(),
            marker_color=TIER_COLORS[tier],
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in pivot[tier].tolist()],
            textposition="inside",
            textfont=dict(size=9, family="Space Mono, monospace"),
            hovertemplate=f"T{tier} · %{{x}}<br>%{{y:.1f}}% of group<extra></extra>",
        ))
    fig.update_layout(
        **CHART_BASE, height=250, barmode="stack",
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", showline=False,
                   title=dict(text="% of Group", font=dict(size=11)),
                   range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

f1, f2, f3 = st.columns(3, gap="medium")

with f1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Tier Distribution by Gender",
                "Stacked 100% bar — equal tier share = no gender bias")
    sex_pivot = _tier_pct_pivot("SEX", {1: "Male", 2: "Female"})
    st.plotly_chart(_fairness_bar(sex_pivot, "sex"), use_container_width=True,
                    config={"displayModeBar": False}, key="fair_sex")
    male_t5   = sex_pivot.loc["Male",   5] if "Male"   in sex_pivot.index and 5 in sex_pivot.columns else 0
    female_t5 = sex_pivot.loc["Female", 5] if "Female" in sex_pivot.index and 5 in sex_pivot.columns else 0
    gap_color = "#dc2626" if abs(male_t5 - female_t5) > 2 else "#16a34a"
    st.markdown(
        f'<p style="font-size:11px;color:{gap_color};font-weight:700;margin-top:8px;letter-spacing:0.01em;">'
        f'Tier 5 gap — Male: {male_t5:.1f}% vs Female: {female_t5:.1f}%'
        f' ({"flag: review" if abs(male_t5 - female_t5) > 2 else "within tolerance"})</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with f2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Tier Distribution by Age Group",
                "Binned into 4 cohorts — checks for age-based escalation bias")
    age_df = df.copy()
    age_df["age_group"] = pd.cut(
        age_df["AGE"], bins=[0, 30, 40, 50, 200],
        labels=["<30", "30–39", "40–49", "50+"]
    )
    age_pivot = (age_df.groupby(["age_group", "recovery_tier_final"], observed=False)
                 .size().unstack(fill_value=0))
    age_pct = age_pivot.div(age_pivot.sum(axis=1), axis=0) * 100
    st.plotly_chart(_fairness_bar(age_pct, "age"), use_container_width=True,
                    config={"displayModeBar": False}, key="fair_age")
    if 5 in age_pct.columns:
        max_age_t5 = age_pct[5].idxmax()
        st.markdown(
            f'<p style="font-size:11px;color:#94a3b8;font-weight:500;margin-top:8px;">'
            f'Highest Tier 5 rate: <b>{max_age_t5}</b> age group '
            f'({age_pct.loc[max_age_t5, 5]:.1f}%)</p>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with f3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("Tier Distribution by Education",
                "1=Graduate · 2=University · 3=High School · 4=Other")
    edu_pivot = _tier_pct_pivot(
        "EDUCATION",
        {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
    )
    st.plotly_chart(_fairness_bar(edu_pivot, "edu"), use_container_width=True,
                    config={"displayModeBar": False}, key="fair_edu")
    if 5 in edu_pivot.columns:
        max_edu_t5 = edu_pivot[5].idxmax()
        st.markdown(
            f'<p style="font-size:11px;color:#94a3b8;font-weight:500;margin-top:8px;">'
            f'Highest Tier 5 rate: <b>{max_edu_t5}</b> '
            f'({edu_pivot.loc[max_edu_t5, 5]:.1f}%)</p>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
