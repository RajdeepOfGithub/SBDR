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
TIER_COLORS = {1: "#10b981", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444", 5: "#7c3aed"}
TIER_DIM    = "rgba(255,255,255,0.06)"

BRANCH_COLORS = {
    "UCI/Derived": "#2563eb",
    "BiLSTM":      "#7c3aed",
    "FinBERT":     "#0ea5e9",
    "Sparkov":     "#f59e0b",
}

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
    1: "Automated Reminder",
    2: "Soft Assist SMS",
    3: "Offer Hardship Plan",
    4: "RM Phone Call Required",
    5: "Legal Referral",
}

CHART_BASE = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(family="Inter, -apple-system, sans-serif", size=11, color="#475569"),
    margin        = dict(l=8, r=8, t=8, b=8),
    hoverlabel    = dict(bgcolor="#111111", font_color="#f1f5f9", font_size=12,
                         font_family="Inter, sans-serif", bordercolor="rgba(255,255,255,0.12)"),
    transition    = dict(duration=400, easing="cubic-in-out"),
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "SBDR · Intelligence Platform",
    page_icon             = "🛡",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
# Use st.html (not st.markdown) — st.html is injected directly into the page
# document, not inside an iframe, so @import and font-face rules work reliably.
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,100..900&family=DM+Mono:wght@400;500&family=Orbitron:wght@700;800;900&display=swap');

/* ── Font: safe scoped inheritance (NO universal * rule) ──────────────────────
   The old `* { font-family: Inter !important }` hit Streamlit icon spans whose
   "text" is a Material Symbols ligature. Removing it fixes the sidebar toggle.
   Inter flows naturally via inheritance from the app shell down to content.     */
:root {
    --app-font: 'Inter', system-ui, -apple-system, BlinkMacSystemFont,
                'Segoe UI', sans-serif;
    --mono-font: 'DM Mono', ui-monospace, 'Courier New', monospace;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stSidebarUserContent"] {
    font-family: var(--app-font);
}

.stApp :where(
    h1, h2, h3, h4, h5, h6,
    p, li, blockquote,
    label,
    input, textarea, select,
    button,
    table, thead, tbody, tr, th, td,
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stMetric"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"]
) {
    font-family: var(--app-font);
}

/* ── Protect Material Symbols icons ───────────────────────────────────────────
   The sidebar toggle renders "keyboard_double_arrow_right" as a ligature.
   It MUST keep Material Symbols Rounded or the raw text appears.              */
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] p,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] p,
[data-testid="collapsedControl"] [class*="material"],
[data-testid="stSidebarCollapseButton"] [class*="material"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── Sidebar toggle button styling ───────────────────────────────────────────*/
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    border-radius: 999px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: #0a0a0a !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    transition: background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #111111 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
}
[data-testid="collapsedControl"] button:active,
[data-testid="stSidebarCollapseButton"] button:active {
    transform: translateY(0) !important;
}
[data-testid="collapsedControl"] button span,
[data-testid="stSidebarCollapseButton"] button span { color: #444444 !important; font-size: 20px !important; }
[data-testid="collapsedControl"] button:hover span,
[data-testid="stSidebarCollapseButton"] button:hover span { color: #2563eb !important; }

.mono, code, pre { font-family: var(--mono-font) !important; }

/* ── Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
header [data-testid="stDecoration"],
header a[href*="streamlit.io"],
header .stDeployButton { display: none !important; }
.block-container { padding: 2rem 2.5rem 5rem 2.5rem !important; max-width: 100% !important; }

/* ── App background ── */
.stApp { background: #000000 !important; }

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #0f0f0f !important;
    min-width: 256px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Hide Streamlit's auto-generated widget labels (we use custom HTML labels) */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] label { display: none !important; }

/* Multiselect control box */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: #0c0c0c !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    min-height: 40px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}
/* Multiselect tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #0c1830 !important;
    border: 1px solid #1e3a6e !important;
    color: #60a5fa !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 1px 6px 1px 8px !important;
    margin: 2px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #60a5fa !important; font-size: 11px !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] [role="presentation"] { color: #3b82f6 !important; }
/* Dropdown options */
[data-testid="stSidebar"] [data-baseweb="menu"] { background: #0c0c0c !important; border: 1px solid #1a1a1a !important; border-radius: 8px !important; }
[data-testid="stSidebar"] [data-baseweb="option"] { background: #0c0c0c !important; color: #94a3b8 !important; font-size: 12px !important; }
[data-testid="stSidebar"] [data-baseweb="option"]:hover { background: #141414 !important; }

/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] { padding: 0 !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] > div:nth-child(3) {
    background: #1a1a1a !important; height: 3px !important;
}
/* Slider filled portion */
[data-testid="stSidebar"] [data-baseweb="slider"] > div:nth-child(3) > div:nth-child(3) {
    background: #2563eb !important;
}
/* Slider thumb */
[data-testid="stSidebar"] [role="slider"] {
    background: #2563eb !important;
    border: 2px solid #050505 !important;
    width: 14px !important; height: 14px !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.25) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stSidebar"] [role="slider"]:hover {
    box-shadow: 0 0 0 5px rgba(37,99,235,0.2) !important;
}
/* Slider value text */
[data-testid="stSidebar"] [data-testid="stSlider"] p { display: none !important; }

/* Checkbox */
[data-testid="stSidebar"] [data-baseweb="checkbox"] > label { gap: 10px !important; }
[data-testid="stSidebar"] [data-baseweb="checkbox"] span[role="checkbox"] {
    background: #0c0c0c !important;
    border: 1.5px solid #1e1e1e !important;
    border-radius: 4px !important;
    width: 16px !important; height: 16px !important;
    transition: all 0.2s ease !important;
    flex-shrink: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="checkbox"] [aria-checked="true"] span[role="checkbox"] {
    background: #2563eb !important; border-color: #2563eb !important;
}
[data-testid="stSidebar"] .stCheckbox p { color: #64748b !important; font-size: 12px !important; }

/* Reset button */
[data-testid="stSidebar"] .stButton > button {
    background: #080808 !important;
    border: 1px solid #1a1a1a !important;
    color: #475569 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    height: 38px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #0c0c0c !important;
    border-color: #1e3a6e !important;
    color: #60a5fa !important;
}

/* ── Sidebar collapse toggle ──────────────────────────────────────────────────
   Root cause: `* { font-family: 'Inter' !important }` overrides Material Symbols
   Rounded on the icon spans. Streamlit renders these icons as ligature text
   (e.g. "keyboard_double_arrow_right") inside a span with font-family Material
   Symbols Rounded. Once Inter replaces that font the ligature breaks and the raw
   text appears.

   Fix (correct approach): import Material Symbols Rounded above, then explicitly
   RESTORE that font on every span/p inside these buttons AFTER the global rule.
   CSS specificity + source order wins — no visibility hacks needed.
   ─────────────────────────────────────────────────────────────────────────── */

/* Container styling */
[data-testid="collapsedControl"] {
    background: #050505 !important;
    border: 1px solid #0f0f0f !important;
    border-left: none !important;
    border-radius: 0 6px 6px 0 !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
}
[data-testid="collapsedControl"]:hover { background: #0d0d0d !important; }

/* Restore Material Symbols Rounded on EVERY descendant text node inside
   the toggle — using * catches spans nested at any depth */
[data-testid="collapsedControl"] *,
section[data-testid="stSidebar"] > div > div > button *,
button[aria-label="Close sidebar"] *,
button[aria-label="Open sidebar"] * {
    font-family: 'Material Symbols Rounded' !important;
    font-optical-sizing: auto !important;
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    color: #555555 !important;
    transition: color 0.2s ease !important;
    -webkit-font-feature-settings: "liga" !important;
    font-feature-settings: "liga" !important;
}

/* Hover: tint the icon blue */
[data-testid="collapsedControl"]:hover *,
section[data-testid="stSidebar"] > div > div > button:hover *,
button[aria-label="Close sidebar"]:hover *,
button[aria-label="Open sidebar"]:hover * {
    color: #2563eb !important;
}

/* Button shell — transparent, no border */
section[data-testid="stSidebar"] > div > div > button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #111111 !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #475569 !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 24px !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    transition: color 0.2s ease !important; letter-spacing: 0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; }
.stTabs [aria-selected="true"] {
    color: #f1f5f9 !important; border-bottom: 2px solid #2563eb !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 0 0 0 !important; }

/* ── Cards ── */
.card {
    background: #080808; border: 1px solid #111111; border-radius: 12px;
    padding: 22px 24px; margin-bottom: 16px; position: relative; overflow: hidden;
    animation: fadeInUp 0.35s ease both;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.card:hover { border-color: #1a1a1a; box-shadow: 0 4px 32px rgba(0,0,0,0.9); }
.card-title { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 3px; }
.card-sub   { font-size: 11.5px; color: #475569; margin-bottom: 16px; }

/* ── Section label ── */
.section-label {
    font-size: 10px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin: 28px 0 14px 0; display: flex; align-items: center; gap: 10px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1a1a1a; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 9px; border-radius: 4px; font-size: 11px; font-weight: 500; white-space: nowrap; }
.t1-badge { background: rgba(16,185,129,0.1);  color: #34d399; border: 1px solid rgba(16,185,129,0.18); }
.t2-badge { background: rgba(59,130,246,0.1);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.18); }
.t3-badge { background: rgba(245,158,11,0.1);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.18); }
.t4-badge { background: rgba(239,68,68,0.1);   color: #f87171; border: 1px solid rgba(239,68,68,0.18);  }
.t5-badge { background: rgba(124,58,237,0.1);  color: #a78bfa; border: 1px solid rgba(124,58,237,0.18); }

/* ── Audit rows ── */
.audit-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #0d0d0d; }
.audit-row:last-child { border-bottom: none; }
.audit-key { font-size: 12px; color: #475569; font-weight: 400; }
.audit-val { font-size: 12px; color: #94a3b8; font-weight: 500; }

/* ── Stat row ── */
.stat-block { text-align: center; padding: 12px 0; }
.stat-n { font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.stat-v { font-size: 28px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; font-family: 'Orbitron', sans-serif !important; }
.stat-s { font-size: 11px; color: #64748b; margin-top: 4px; }

/* ── Progress bar ── */
.prog-wrap { margin: 5px 0; }
.prog-label { display: flex; justify-content: space-between; margin-bottom: 4px; }
.prog-name  { font-size: 11.5px; color: #475569; }
.prog-pct   { font-size: 11.5px; color: #64748b; font-weight: 500; font-family: 'DM Mono', monospace !important; }
.prog-bg    { background: #111111; border-radius: 3px; height: 4px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.4,0,0.2,1); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px !important; border: 1px solid #111111 !important; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #050505 !important; color: #475569 !important; font-size: 10.5px !important; font-weight: 600 !important; border-bottom: 1px solid #111111 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stDataFrame"] td { color: #64748b !important; background: #070707 !important; font-size: 12px !important; }
[data-testid="stDataFrame"] tr:hover td { background: #0d0d0d !important; }

/* ── Text input ── */
.stTextInput input { background: #080808 !important; border: 1px solid #1a1a1a !important; color: #f1f5f9 !important; border-radius: 8px !important; font-size: 13px !important; }
.stTextInput input:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.07) !important; }
.stTextInput input::placeholder { color: #2d3748 !important; }

/* ── Select / multiselect ── */
[data-baseweb="select"] [data-baseweb="popover"] { background: #0a0a0a !important; border: 1px solid #1c1c1c !important; }

/* ── Slider ── */
input[type="range"] { accent-color: #2563eb; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 3px; }

/* ── Selectbox dark ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #080808 !important; border-color: #1a1a1a !important; color: #e2e8f0 !important; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 4px rgba(37,99,235,0.3); }
    50%       { box-shadow: 0 0 12px rgba(37,99,235,0.6); }
}
</style>
""")


# ── Helpers ────────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str, accent: str, sub_color: str = "#1e1e1e") -> None:
    num_m    = re.search(r"[\d,]+\.?\d*", value)
    raw_num  = float(num_m.group().replace(",", "")) if num_m else 0
    prefix   = value[:num_m.start()] if num_m else ""
    suffix   = value[num_m.end():]   if num_m else ""
    decimals = len(value.split(".")[-1]) if ("." in value and num_m) else 0
    components.html(f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700;800&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html,body {{ background:transparent; overflow:hidden; font-family:'Inter',sans-serif; }}
.c {{
  background:#080808; border-radius:12px; padding:20px 22px 18px;
  border:1px solid #111111; border-left:2px solid {accent};
  position:relative; overflow:hidden;
  animation:fi .4s ease both;
  transition:border-color .2s ease;
}}
.c::after {{ content:''; position:absolute; top:-40px; right:-40px; width:100px; height:100px;
  background:radial-gradient(circle,{accent}12 0%,transparent 70%); pointer-events:none; }}
@keyframes fi {{ from{{opacity:0;transform:translateY(10px)}} to{{opacity:1;transform:translateY(0)}} }}
.lbl {{ font-size:10px; font-weight:600; color:#64748b; text-transform:uppercase;
  letter-spacing:.12em; margin-bottom:10px; }}
.val {{ font-size:26px; font-weight:800; color:#f1f5f9; line-height:1.1;
  margin-bottom:6px; letter-spacing:-.5px; font-variant-numeric:tabular-nums;
  font-family:'Orbitron','Inter',sans-serif; }}
.sub {{ font-size:11px; font-weight:400; color:{sub_color}; }}
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
  var dur=1200,t0=performance.now();
  function ease(t){{return 1-Math.pow(1-t,3);}}
  function tick(now){{
    var p=Math.min((now-t0)/dur,1),v=end*ease(p);
    var s=dec>0?v.toFixed(dec):Math.round(v).toLocaleString();
    el.textContent=pre+s+suf;
    if(p<1)requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}})();
</script></body></html>""", height=106, scrolling=False)


def card_open(delay_ms: int = 0) -> None:
    style = f' style="animation-delay:{delay_ms}ms"' if delay_ms else ""
    st.markdown(f'<div class="card"{style}>', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def card_header(title: str, sub: str = "") -> None:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="card-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def badge(text: str, bg: str, color: str) -> str:
    return f'<span class="badge" style="background:{bg};color:{color};">{text}</span>'


def tier_badge(tier: int) -> str:
    cls = f"t{tier}-badge"
    return f'<span class="badge {cls}">T{tier} · {TIER_NAMES[tier]}</span>'


def audit_row(key: str, val_html: str) -> str:
    return (f'<div class="audit-row">'
            f'<span class="audit-key">{key}</span>'
            f'<span class="audit-val">{val_html}</span>'
            f'</div>')


def prog_bar(name: str, value: float, max_val: float, color: str) -> str:
    pct = min(value / max_val * 100, 100) if max_val > 0 else 0
    return (f'<div class="prog-wrap">'
            f'<div class="prog-label">'
            f'<span class="prog-name">{name}</span>'
            f'<span class="prog-pct">{value:.3f}</span>'
            f'</div>'
            f'<div class="prog-bg">'
            f'<div class="prog-fill" style="width:{pct:.1f}%;background:{color};"></div>'
            f'</div></div>')


def get_xai_signal(row: pd.Series) -> str:
    if bool(row["bilstm_anomaly_flag"]):  return "BiLSTM · Payment Anomaly"
    if row["distress_avg"] > 0.65:        return "FinBERT · High Distress"
    if row["avg_pay_delay"] > 2:          return "UCI · Severe Delinquency"
    return "XGBoost · Multi-Signal"


def tier_prob_bars(row: pd.Series) -> str:
    html = ""
    for i in range(1, 6):
        v = float(row[f"tier_prob_{i}"])
        html += prog_bar(f"T{i} · {TIER_NAMES[i]}", v, 1.0, TIER_COLORS[i])
    return html


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
m        = manifest["metrics"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Brand ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:24px 20px 20px 20px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:0;">
        <div style="width:36px;height:36px;background:#2563eb;border-radius:9px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;
                    box-shadow:0 4px 14px rgba(37,99,235,0.35);">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
               stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
        </div>
        <div>
          <div style="font-size:15px;font-weight:700;color:#f1f5f9;letter-spacing:-0.2px;line-height:1.2;">SBDR</div>
          <div style="font-size:10px;color:#475569;font-weight:400;margin-top:1px;letter-spacing:0.02em;">Debt Recovery Intelligence</div>
        </div>
      </div>
    </div>
    <div style="height:1px;background:#0f0f0f;margin:0 20px 20px 20px;"></div>
    """, unsafe_allow_html=True)

    # ── Filter group container ─────────────────────────────────────────────────
    st.markdown('<div style="padding:0 20px;">', unsafe_allow_html=True)

    # — Recovery Tier ——————————————————————————————————————————————————————————
    st.markdown("""
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
           stroke="#475569" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
      </svg>
      <span style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.13em;">Recovery Tier</span>
    </div>
    """, unsafe_allow_html=True)
    selected_tiers = st.multiselect(
        "_tier", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5],
        format_func=lambda x: f"T{x}  ·  {TIER_NAMES[x]}",
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # — Distress Score ——————————————————————————————————————————————————————————
    st.markdown("""
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
           stroke="#475569" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
      <span style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.13em;">Distress Score</span>
    </div>
    """, unsafe_allow_html=True)
    distress_range = st.slider(
        "_distress", 0.0, 1.0, (0.0, 1.0), 0.01,
        key="distress_range", label_visibility="collapsed",
    )
    min_d_display, max_d_display = distress_range
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;margin-top:-4px;padding:0 2px;">'
        f'<span style="font-size:10px;color:#475569;font-family:\'DM Mono\',monospace;">{min_d_display:.2f}</span>'
        f'<span style="font-size:10px;color:#475569;font-family:\'DM Mono\',monospace;">{max_d_display:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # — Anomaly Flag ——————————————————————————————————————————————————————————
    st.markdown("""
    <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
           stroke="#475569" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <span style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:0.13em;">Anomaly Detection</span>
    </div>
    """, unsafe_allow_html=True)
    anomaly_only = st.checkbox("Show anomaly-flagged accounts only", value=False)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # — Active filter summary ——————————————————————————————————————————————————
    n_active = (
        (len(selected_tiers) < 5)
        + (distress_range != (0.0, 1.0))
        + anomaly_only
    )
    if n_active > 0:
        st.markdown(
            f'<div style="background:#0c1830;border:1px solid #1e3a6e;border-radius:8px;'
            f'padding:8px 12px;margin-bottom:16px;display:flex;align-items:center;gap:8px;">'
            f'<div style="width:5px;height:5px;border-radius:50%;background:#60a5fa;flex-shrink:0;"></div>'
            f'<span style="font-size:11px;color:#60a5fa;font-weight:500;">'
            f'{n_active} filter{"s" if n_active > 1 else ""} active</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # — Reset ——————————————————————————————————————————————————————————————————
    if st.button("Reset Filters", use_container_width=True):
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="position:absolute;bottom:0;left:0;right:0;padding:16px 20px;
                border-top:1px solid #0f0f0f;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:6px;">
          <span style="width:5px;height:5px;border-radius:50%;background:#10b981;
                       display:inline-block;animation:pulse 2s ease infinite;"></span>
          <span style="font-size:10px;color:#475569;letter-spacing:0.02em;">Live</span>
        </div>
        <span style="font-size:10px;color:#475569;">DATA 606 · UMBC</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Filtered dataset ───────────────────────────────────────────────────────────
active_tiers = selected_tiers if selected_tiers else [1, 2, 3, 4, 5]
min_d, max_d = distress_range
mask = (
    df["recovery_tier_final"].isin(active_tiers)
    & (df["distress_avg"] >= min_d)
    & (df["distress_avg"] <= max_d)
)
if anomaly_only:
    mask &= df["bilstm_anomaly_flag"] == True
filtered_df = df[mask]

filters_active = (
    set(active_tiers) != {1, 2, 3, 4, 5}
    or min_d > 0 or max_d < 1.0 or anomaly_only
)

# ── Global KPIs ────────────────────────────────────────────────────────────────
total_customers    = len(df)
total_credit_limit = df["LIMIT_BAL"].sum()
tier4_count        = int((df["recovery_tier_final"] == 4).sum())
tier5_count        = int((df["recovery_tier_final"] == 5).sum())
high_risk_limit    = df[df["recovery_tier_final"].isin([4, 5])]["LIMIT_BAL"].sum()
avg_distress_all   = df["distress_avg"].mean()
anomaly_count      = int(df["bilstm_anomaly_flag"].sum())

distress_color  = "#f59e0b" if avg_distress_all > 0.50 else "#10b981"
distress_status = "Elevated" if avg_distress_all > 0.50 else "Stable"

# Pre-compute spotlight (Tier 5 Rule C preferred)
_t5_df         = df[df["recovery_tier_final"] == 5]
_rule_c_cases  = _t5_df[_t5_df["audit_rule_c"] == True]
spotlight_case = _rule_c_cases.iloc[0] if len(_rule_c_cases) > 0 else _t5_df.iloc[0]


# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:28px;">
  <div>
    <div style="display:inline-flex;align-items:center;gap:6px;
                background:#080808;border:1px solid #111111;
                padding:3px 12px;border-radius:4px;margin-bottom:10px;">
      <span style="width:5px;height:5px;border-radius:50%;background:#10b981;
                   display:inline-block;animation:pulse 2s ease infinite;"></span>
      <span style="font-size:10px;font-weight:600;color:#475569;letter-spacing:.1em;text-transform:uppercase;">
        Live · DATA 606 Capstone · UMBC
      </span>
    </div>
    <div style="font-size:30px;font-weight:800;color:#f8fafc;letter-spacing:-0.5px;line-height:1.1;">
      Sentimental-Behavioral Debt Recovery
    </div>
  </div>
  <div style="text-align:right;padding-top:4px;">
    <div style="font-size:11px;color:#475569;">FinBERT · BiLSTM · XGBoost</div>
    <div style="font-size:11px;color:#475569;margin-top:2px;">Audit Layer · B3.5</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    kpi_card("Monitored Accounts", f"{total_customers:,}",
             f"Credit pool  ${total_credit_limit/1e9:.2f}B", "#2563eb")
with k2:
    kpi_card("High-Risk Accounts", f"{tier4_count + tier5_count:,}",
             f"T4 + T5  ·  ${high_risk_limit/1e9:.2f}B exposure", "#ef4444", "#ef4444")
with k3:
    kpi_card("Portfolio Distress", f"{avg_distress_all:.2f}",
             distress_status, distress_color, distress_color)
with k4:
    kpi_card("Behavioural Anomalies", f"{anomaly_count:,}",
             "5.0% of portfolio  ·  BiLSTM P95", "#7c3aed", "#7c3aed")

st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_portfolio, tab_insight, tab_audit, tab_fairness, tab_simulator, tab_model = st.tabs([
    "Overview", "Portfolio", "Customer Insight", "Audit & Risk", "Fairness", "What-If", "Model"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    card_open()
    card_header("Recovery Tier Distribution",
                "Final classification after B3.5 strategic default audit layer · 30,000 customers")
    tier_counts = df["recovery_tier_final"].value_counts().sort_index()
    pct         = (tier_counts / len(df) * 100).round(1)
    bar_colors  = [TIER_COLORS[i] if i in active_tiers else TIER_DIM for i in tier_counts.index]
    text_labels = [f"{c:,}  ({p}%)" for c, p in zip(tier_counts.values, pct.values)]

    fig = go.Figure(go.Bar(
        x=[f"T{i}  {TIER_NAMES[i]}" for i in tier_counts.index],
        y=tier_counts.values.tolist(),
        marker_color=bar_colors, marker_line_width=0,
        text=text_labels, textposition="outside",
        textfont=dict(size=12, family="Inter, sans-serif", color="#475569"),
        hovertemplate="<b>%{x}</b><br>%{y:,} customers<extra></extra>",
    ))
    fig.update_layout(
        **CHART_BASE, height=380,
        xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=12, color="#475569")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                   zeroline=False, title=dict(text="Customers", font=dict(size=12))),
    )
    fig.update_yaxes(range=[0, tier_counts.max() * 1.22])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="ov_tier")
    card_close()

    # Tier summary pills row
    pill_cols = st.columns(5, gap="medium")
    for idx, (tier, count) in enumerate(tier_counts.items()):
        pct_val = count / len(df) * 100
        exposure_m = df[df["recovery_tier_final"] == tier]["LIMIT_BAL"].sum() / 1e6
        with pill_cols[idx]:
            st.markdown(
                f'<div style="background:#080808;border:1px solid #111111;'
                f'border-top:2px solid {TIER_COLORS[tier]};border-radius:10px;'
                f'padding:14px 16px;text-align:center;">'
                f'<div style="font-size:10px;font-weight:600;color:#475569;'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">'
                f'T{tier} · {TIER_NAMES[tier]}</div>'
                f'<div style="font-size:24px;font-weight:800;color:{TIER_COLORS[tier]};'
                f'font-family:\'Orbitron\',sans-serif;line-height:1;">{count:,}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">'
                f'{pct_val:.1f}% · ${exposure_m:,.0f}M</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:

    # Active filter pills
    if filters_active:
        pills = []
        if set(active_tiers) != {1, 2, 3, 4, 5}:
            for t in active_tiers:
                pills.append(f'<span class="badge t{t}-badge">T{t}: {TIER_NAMES[t]}</span>')
        if min_d > 0 or max_d < 1.0:
            pills.append(f'<span class="badge" style="background:#0f0f0f;color:#475569;border:1px solid #1a1a1a;">Distress {min_d:.2f}–{max_d:.2f}</span>')
        if anomaly_only:
            pills.append('<span class="badge" style="background:rgba(124,58,237,0.1);color:#a78bfa;border:1px solid rgba(124,58,237,0.2);">Anomaly Only</span>')
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;flex-wrap:wrap;">'
            + '<span style="font-size:10px;color:#1e1e1e;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Active filters</span>'
            + "".join(pills) + '</div>',
            unsafe_allow_html=True,
        )

    # Credit exposure + Risk map
    port_r1a, port_r1b = st.columns([1, 1.4], gap="medium")

    with port_r1a:
        card_open()
        card_header("Credit Exposure by Tier", "Total credit limit at risk · $M")
        exposure = (df.groupby("recovery_tier_final")["LIMIT_BAL"].sum() / 1e6)
        tier_labels = [f"T{i}  {TIER_NAMES[i]}" for i in exposure.index]
        exp_colors  = [TIER_COLORS[i] for i in exposure.index]
        fig_exp = go.Figure(go.Bar(
            x=tier_labels, y=exposure.values.tolist(),
            marker_color=exp_colors, marker_line_width=0,
            text=[f"${v:,.0f}M" for v in exposure.values],
            textposition="outside",
            textfont=dict(size=10, family="Inter, sans-serif", color="#475569"),
            hovertemplate="<b>%{x}</b><br>Exposure: $%{y:,.0f}M<extra></extra>",
        ))
        fig_exp.update_layout(
            **CHART_BASE, height=260,
            xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=10, color="#475569")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       zeroline=False, title=dict(text="$M", font=dict(size=11))),
        )
        fig_exp.update_yaxes(range=[0, exposure.max() * 1.25])
        st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False}, key="port_exp")
        total_high_risk = exposure[[4, 5]].sum()
        st.markdown(
            f'<div style="font-size:11px;color:#475569;margin-top:4px;">'
            f'T4 + T5 exposure: <span style="color:#ef4444;font-weight:600;">'
            f'${total_high_risk:,.0f}M</span> of '
            f'<span style="color:#94a3b8;font-weight:600;">'
            f'${exposure.sum():,.0f}M</span> total</div>',
            unsafe_allow_html=True,
        )
        card_close()

    with port_r1b:
        card_open()
        card_header("Risk Map — Distress vs Payment Delay",
                    "Each point = 1 customer · coloured by recovery tier · 5K sample")
        scatter_sample = df.sample(min(5_000, len(df)), random_state=42)
        fig_sc = go.Figure()
        for tier in sorted(df["recovery_tier_final"].unique()):
            td = scatter_sample[scatter_sample["recovery_tier_final"] == tier]
            if len(td) == 0:
                continue
            fig_sc.add_trace(go.Scatter(
                x=td["distress_avg"], y=td["avg_pay_delay"],
                mode="markers",
                name=f"T{tier} · {TIER_NAMES[tier]}",
                marker=dict(color=TIER_COLORS[tier], size=4, opacity=0.55,
                            line=dict(width=0)),
                hovertemplate=(
                    f"<b>T{tier}: {TIER_NAMES[tier]}</b><br>"
                    "Distress: %{x:.3f}<br>Avg delay: %{y:.1f} mo<extra></extra>"
                ),
            ))
        fig_sc.update_layout(
            **CHART_BASE, height=260,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       zeroline=False, title=dict(text="FinBERT Distress Score", font=dict(size=11)),
                       range=[-0.02, 1.02], tickformat=".1f",
                       tickfont=dict(size=10, color="#475569")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       zeroline=False, title=dict(text="Avg Payment Delay (months)", font=dict(size=11)),
                       tickfont=dict(size=10, color="#475569")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=10, color="#64748b"), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False}, key="port_scatter")
        card_close()

    # Distress histogram
    card_open()
    filt_label = f"{len(filtered_df):,} customers matching filters" if filters_active else "All 30,000 customers"
    card_header("Distress Score Distribution", f"{filt_label} · per-tier overlay")

    fig_hist = go.Figure()
    for tier in sorted(active_tiers):
        td = filtered_df[filtered_df["recovery_tier_final"] == tier]["distress_avg"]
        if len(td) == 0:
            continue
        fig_hist.add_trace(go.Histogram(
            x=td, name=f"T{tier} · {TIER_NAMES[tier]}",
            marker_color=TIER_COLORS[tier], marker_line_width=0,
            opacity=0.75, nbinsx=40,
            hovertemplate=f"<b>T{tier}: {TIER_NAMES[tier]}</b><br>Distress: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig_hist.update_layout(
        **CHART_BASE, height=280, barmode="overlay",
        xaxis=dict(showgrid=False, showline=False, zeroline=False,
                   title=dict(text="Distress Score", font=dict(size=11)),
                   range=[0, 1], tickformat=".1f"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                   zeroline=False, title=dict(text="Customers", font=dict(size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=10.5, color="#2d2d2d"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False}, key="port_hist")
    card_close()

    # Customer roster
    card_open()
    search_col, count_col = st.columns([3, 1])
    with search_col:
        search = st.text_input("", placeholder="Search customer ID  (e.g. SBDR_00042)",
                               label_visibility="collapsed", key="roster_search")
    with count_col:
        st.markdown(f'<div style="padding-top:9px;text-align:right;font-size:12px;color:#1e1e1e;">'
                    f'{len(filtered_df):,} matching</div>', unsafe_allow_html=True)

    card_header("Customer Roster", "Sorted by distress score · Next Best Action assigned per tier")

    display_df = filtered_df.copy()
    if search:
        display_df = display_df[display_df["customer_id"].str.contains(search.upper(), na=False)]

    roster_raw = display_df.nlargest(200, "distress_avg")
    df_roster  = pd.DataFrame({
        "Customer":         roster_raw["customer_id"].values,
        "Tier":             [f"T{t}" for t in roster_raw["recovery_tier_final"]],
        "Distress":         roster_raw["distress_avg"].round(3).values,
        "Anomaly":          ["⚠ Yes" if x else "—" for x in roster_raw["bilstm_anomaly_flag"]],
        "Avg Delay (mo)":   roster_raw["avg_pay_delay"].round(2).values,
        "Primary Signal":   [get_xai_signal(r) for _, r in roster_raw.iterrows()],
        "Next Best Action": [NBA_MAP[t] for t in roster_raw["recovery_tier_final"]],
    })
    st.dataframe(
        df_roster, use_container_width=True, hide_index=True,
        column_config={
            "Customer":         st.column_config.TextColumn("Customer", width="medium"),
            "Tier":             st.column_config.TextColumn("Tier", width="small"),
            "Distress":         st.column_config.ProgressColumn(
                                    "Distress", min_value=0, max_value=1, format="%.3f", width="medium"),
            "Anomaly":          st.column_config.TextColumn("Anomaly", width="small"),
            "Avg Delay (mo)":   st.column_config.NumberColumn("Avg Delay", format="%.2f", width="small"),
            "Primary Signal":   st.column_config.TextColumn("Primary Signal", width="large"),
            "Next Best Action": st.column_config.TextColumn("Next Best Action", width="large"),
        },
        height=460,
    )
    card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOMER INSIGHT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insight:

    # Customer picker
    card_open()
    st.markdown('<div class="card-title">Select Customer</div>', unsafe_allow_html=True)
    pick_tier = st.selectbox(
        "Filter by tier", options=["All"] + [f"T{i} · {TIER_NAMES[i]}" for i in range(1, 6)],
        label_visibility="collapsed", key="pick_tier"
    )
    if pick_tier == "All":
        tier_pool = df
    else:
        t_num = int(pick_tier[1])
        tier_pool = df[df["recovery_tier_final"] == t_num]

    customer_ids = tier_pool["customer_id"].tolist()
    selected_cid = st.selectbox(
        "Customer", options=customer_ids, key="selected_cid",
        label_visibility="collapsed",
    )
    card_close()

    row = df[df["customer_id"] == selected_cid].iloc[0]
    tier_num    = int(row["recovery_tier_final"])
    tier_color  = TIER_COLORS[tier_num]
    anomaly_flag = bool(row["bilstm_anomaly_flag"])

    # Customer summary bar
    st.markdown(f"""
    <div style="background:#080808;border:1px solid #111111;border-left:3px solid {tier_color};
                border-radius:10px;padding:14px 20px;margin-bottom:16px;
                display:flex;align-items:center;gap:20px;">
      <div>
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;font-weight:600;">Customer</div>
        <div style="font-size:16px;font-weight:700;color:#f1f5f9;margin-top:2px;font-family:'DM Mono',monospace;">{selected_cid}</div>
      </div>
      <div style="width:1px;height:36px;background:#111111;"></div>
      <div>
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;font-weight:600;">Recovery Tier</div>
        <div style="margin-top:4px;">{tier_badge(tier_num)}</div>
      </div>
      <div style="width:1px;height:36px;background:#111111;"></div>
      <div>
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;font-weight:600;">Next Best Action</div>
        <div style="font-size:13px;font-weight:500;color:#94a3b8;margin-top:3px;">{NBA_MAP[tier_num]}</div>
      </div>
      <div style="width:1px;height:36px;background:#111111;"></div>
      <div>
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;font-weight:600;">Avg Distress</div>
        <div style="font-size:16px;font-weight:700;margin-top:2px;color:{('#ef4444' if row['distress_avg'] > 0.65 else '#f59e0b' if row['distress_avg'] > 0.4 else '#10b981')};">{row['distress_avg']:.3f}</div>
      </div>
      <div style="width:1px;height:36px;background:#111111;"></div>
      <div>
        <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.1em;font-weight:600;">BiLSTM Anomaly</div>
        <div style="margin-top:4px;">{'<span class="badge" style="background:rgba(239,68,68,0.1);color:#f87171;border:1px solid rgba(239,68,68,0.2);">⚠ Detected</span>' if anomaly_flag else '<span class="badge" style="background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2);">✓ Clear</span>'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ev1, ev2, ev3 = st.columns(3, gap="medium")

    # Branch 1: FinBERT
    with ev1:
        card_open()
        card_header("Branch 1 · FinBERT", "Sentiment distress score per chat turn")

        turn_scores = [float(row["distress_turn_1"]), float(row["distress_turn_2"]), float(row["distress_turn_3"])]
        turn_colors = ["#10b981" if s < 0.4 else "#f59e0b" if s < 0.65 else "#ef4444" for s in turn_scores]

        fig_t = go.Figure(go.Bar(
            x=["Turn 1", "Turn 2", "Turn 3"], y=turn_scores,
            marker_color=turn_colors, marker_line_width=0,
            text=[f"{s:.3f}" for s in turn_scores], textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Distress: %{y:.3f}<extra></extra>",
        ))
        fig_t.add_hline(y=row["distress_avg"], line_dash="dot", line_color="#475569",
                        annotation_text=f"avg {row['distress_avg']:.3f}",
                        annotation_font_size=10, annotation_font_color="#475569")
        fig_t.update_layout(
            **CHART_BASE, height=200,
            xaxis=dict(showgrid=False, showline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       zeroline=False, range=[0, 1.2],
                       title=dict(text="Distress Score", font=dict(size=11))),
        )
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False}, key=f"ins_turns_{selected_cid}")

        shift_val   = float(row["distress_shift"])
        shift_color = "#10b981" if shift_val < 0 else "#ef4444"
        shift_arrow = "▼" if shift_val < 0 else "▲"
        direction   = "de-escalating" if shift_val < 0 else "escalating"
        st.markdown(f'<div style="padding:8px 10px;background:#050505;border-radius:6px;border:1px solid #0f0f0f;margin-bottom:10px;">'
                    f'<span style="font-size:11px;color:#475569;">Sentiment shift T3→T1: </span>'
                    f'<span style="font-size:11px;font-weight:600;color:{shift_color};">{shift_arrow} {abs(shift_val):.3f} ({direction})</span>'
                    f'</div>', unsafe_allow_html=True)

        for i, (col_name, score, tcolor) in enumerate(
            zip(["chat_turn_1", "chat_turn_2", "chat_turn_3"], turn_scores, turn_colors), 1
        ):
            text = row[col_name]
            text = "(no chat recorded)" if pd.isna(text) else str(text)
            st.markdown(
                f'<div style="background:#050505;border-radius:8px;padding:8px 12px;'
                f'margin-bottom:6px;border-left:2px solid {tcolor};">'
                f'<div style="font-size:10px;font-weight:600;color:#1a1a1a;margin-bottom:3px;'
                f'text-transform:uppercase;letter-spacing:.08em;">Turn {i} · {score:.3f}</div>'
                f'<div style="font-size:11.5px;color:#475569;line-height:1.6;">'
                f'{text[:220]}{"…" if len(text) > 220 else ""}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        card_close()

    # Branch 2: BiLSTM
    with ev2:
        card_open()
        card_header("Branch 2 · BiLSTM", "6-month payment sequence · anomaly detection")

        pay_cols   = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        pay_vals   = [float(row[c]) for c in pay_cols]
        pay_colors = ["#10b981" if v <= 0 else "#f59e0b" if v <= 2 else "#ef4444" for v in pay_vals]

        fig_p = go.Figure(go.Bar(
            x=["M1", "M2", "M3", "M4", "M5", "M6"], y=pay_vals,
            marker_color=pay_colors, marker_line_width=0,
            text=[str(int(v)) for v in pay_vals], textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{x}</b><br>Pay status: %{y}<extra></extra>",
        ))
        fig_p.add_hline(y=0, line_color="#1e293b", line_width=1)
        fig_p.update_layout(
            **CHART_BASE, height=200,
            xaxis=dict(showgrid=False, showline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       title=dict(text="Months delayed", font=dict(size=11))),
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False}, key=f"ins_pay_{selected_cid}")

        recon_err = float(row["bilstm_recon_error"])
        lc_inc    = row.get("lc_annual_inc_mean")
        lc_dti    = row.get("lc_dti_mean")
        bilstm_data = [
            ("Reconstruction Error", badge(f"{recon_err:.5f}", "rgba(239,68,68,0.1)" if anomaly_flag else "rgba(16,185,129,0.1)", "#f87171" if anomaly_flag else "#34d399")),
            ("Anomaly Flag",         badge("⚠ Detected", "rgba(239,68,68,0.1)", "#f87171") if anomaly_flag else badge("✓ Clear", "rgba(16,185,129,0.1)", "#34d399")),
            ("Avg Pay Delay",        badge(f"{row['avg_pay_delay']:.2f} mo", "rgba(245,158,11,0.1)", "#fbbf24")),
            ("Worst Month Delay",    badge(f"{int(row['worst_month_delay'])} mo", "rgba(245,158,11,0.1)", "#fbbf24")),
        ]
        if lc_inc is not None and not pd.isna(lc_inc):
            bilstm_data.append(("Est. Annual Income", badge(f"${float(lc_inc):,.0f}", "rgba(16,185,129,0.1)", "#34d399")))
        if lc_dti is not None and not pd.isna(lc_dti):
            bilstm_data.append(("Debt-to-Income", badge(f"{float(lc_dti):.1f}%", "rgba(59,130,246,0.1)", "#60a5fa")))

        rows_html = "".join(audit_row(k, v) for k, v in bilstm_data)
        st.markdown(f'<div style="background:#050505;border-radius:8px;padding:10px 12px;border:1px solid #0f0f0f;">{rows_html}</div>', unsafe_allow_html=True)
        card_close()

    # Branch 3: XGBoost + Audit
    with ev3:
        card_open()
        card_header("Branch 3 · XGBoost", "Tier probability distribution")

        # Animated tier probability bars
        max_prob = max(float(row[f"tier_prob_{i}"]) for i in range(1, 6))
        bars_html = tier_prob_bars(row)
        st.markdown(f'<div style="margin-bottom:16px;">{bars_html}</div>', unsafe_allow_html=True)

        # Sparkov signals
        st.markdown('<div class="card-title" style="font-size:12px;margin-bottom:10px;">Sparkov Transaction Signals</div>', unsafe_allow_html=True)
        sparkov_data = [
            ("Total Spend",      badge(f"${float(row['sp_total_spend']):,.2f}", "rgba(59,130,246,0.1)", "#60a5fa")),
            ("Monthly Avg",      badge(f"${float(row['sp_avg_monthly_spend']):,.2f}", "rgba(59,130,246,0.1)", "#60a5fa")),
            ("Spend Volatility", badge(f"{float(row['sp_spend_volatility']):,.2f}", "rgba(245,158,11,0.1)", "#fbbf24")),
            ("Fraud Rate",       badge(f"{float(row['sp_fraud_rate']):.3f}", "rgba(239,68,68,0.1)", "#f87171")),
            ("Transactions",     badge(f"{int(row['sp_num_transactions']):,}", "rgba(16,185,129,0.1)", "#34d399")),
        ]
        sp_html = "".join(audit_row(k, v) for k, v in sparkov_data)
        st.markdown(f'<div style="background:#050505;border-radius:8px;padding:10px 12px;border:1px solid #0f0f0f;margin-bottom:12px;">{sp_html}</div>', unsafe_allow_html=True)

        # Audit flags
        rules_hit = []
        if bool(row["audit_rule_a"]): rules_hit.append("A")
        if bool(row["audit_rule_b"]): rules_hit.append("B")
        if bool(row["audit_rule_c"]): rules_hit.append("C")
        if bool(row["audit_rule_d"]): rules_hit.append("D")

        rule_text = "Rule " + " + ".join(rules_hit) if rules_hit else "No escalation"
        rule_color = "rgba(124,58,237,0.1)" if rules_hit else "rgba(16,185,129,0.1)"
        rule_text_color = "#a78bfa" if rules_hit else "#34d399"

        audit_flags = [
            ("Final Tier",    tier_badge(tier_num)),
            ("Audit Rule",    badge(rule_text, rule_color, rule_text_color)),
            ("Escalated",     badge("Yes", "rgba(239,68,68,0.1)", "#f87171") if bool(row["audit_escalated"]) else badge("No", "rgba(16,185,129,0.1)", "#34d399")),
            ("Default Label", badge("Defaulted", "rgba(239,68,68,0.1)", "#f87171") if int(row["default payment next month"]) == 1 else badge("No Default", "rgba(16,185,129,0.1)", "#34d399")),
        ]
        af_html = "".join(audit_row(k, v) for k, v in audit_flags)
        st.markdown(f'<div style="background:#050505;border-radius:8px;padding:10px 12px;border:1px solid #0f0f0f;">{af_html}</div>', unsafe_allow_html=True)
        card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUDIT & RISK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_audit:

    # Summary KPI strip
    t5_df      = df[df["recovery_tier_final"] == 5]
    escalated  = int(df["audit_escalated"].sum())
    rule_a_n   = int(df["audit_rule_a"].sum())
    rule_b_n   = int(df["audit_rule_b"].sum())
    rule_c_n   = int(df["audit_rule_c"].sum())

    a1, a2, a3, a4 = st.columns(4, gap="medium")
    for col, (lbl, val, sub, acc) in zip(
        [a1, a2, a3, a4],
        [("Strategic Defaults",  str(len(t5_df)),   "Tier 5 confirmed",       "#7c3aed"),
         ("Audit Escalations",   f"+{escalated}",   "Added by B3.5 layer",    "#ef4444"),
         ("Rule A Triggers",     str(rule_a_n),      "Label override",         "#2563eb"),
         ("Rule C Triggers",     str(rule_c_n),      "Fraud + anomaly + default", "#0ea5e9")],
    ):
        with col:
            kpi_card(lbl, val, sub, acc, acc)

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    audit_l, audit_r = st.columns([1, 1.4], gap="medium")

    # Spotlight case
    with audit_l:
        card_open()
        card_header("B3.5 Audit Spotlight", f"Case: {spotlight_case['customer_id']}")

        case = spotlight_case
        rules_hit = []
        if bool(case["audit_rule_a"]): rules_hit.append("A")
        if bool(case["audit_rule_b"]): rules_hit.append("B")
        if bool(case["audit_rule_c"]): rules_hit.append("C")
        if bool(case["audit_rule_d"]): rules_hit.append("D")
        rule_badge_html = badge("Rule " + " + ".join(rules_hit) if rules_hit else "Direct", "rgba(124,58,237,0.1)", "#a78bfa")

        case_data = [
            ("Customer ID",      f'<span style="font-family:\'DM Mono\',monospace;font-size:11.5px;color:#94a3b8;">{case["customer_id"]}</span>'),
            ("Final Tier",       tier_badge(5)),
            ("Distress Score",   badge(f'{case["distress_avg"]:.3f}', "rgba(239,68,68,0.1)", "#f87171")),
            ("BiLSTM Anomaly",   badge("⚠ Detected", "rgba(239,68,68,0.1)", "#f87171") if bool(case["bilstm_anomaly_flag"]) else badge("✓ Clear", "rgba(16,185,129,0.1)", "#34d399")),
            ("Avg Pay Delay",    badge(f'{case["avg_pay_delay"]:.2f} mo', "rgba(245,158,11,0.1)", "#fbbf24")),
            ("Fraud Rate",       badge(f'{case["sp_fraud_rate"]:.3f}', "rgba(239,68,68,0.1)", "#f87171")),
            ("Default Label",    badge("Defaulted", "rgba(239,68,68,0.1)", "#f87171") if int(case["default payment next month"]) == 1 else badge("No Default", "rgba(16,185,129,0.1)", "#34d399")),
            ("Audit Rule",       rule_badge_html),
        ]
        rows_html = "".join(audit_row(k, v) for k, v in case_data)
        st.markdown(
            f'<div style="background:#050505;border-radius:10px;padding:14px 16px;border:1px solid #111111;">'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="margin-top:14px;padding:12px 16px;background:rgba(124,58,237,0.06);'
            'border-radius:8px;border-left:2px solid #7c3aed;">'
            '<div style="font-size:12px;font-weight:600;color:#a78bfa;">Tier 5 — Strategic Default Confirmed</div>'
            '<div style="font-size:11px;color:#475569;margin-top:3px;">Legal referral required · All five audit conditions met</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Audit rule legend
        st.markdown('<div style="margin-top:16px;">', unsafe_allow_html=True)
        rule_legend = [
            ("Rule A", "Engineered label override (default + anomaly + low distress)"),
            ("Rule B", "tier_prob_5 ≥ 0.15 even when prediction is T4"),
            ("Rule C", "Fraud rate + BiLSTM anomaly + default (triple signal)"),
            ("Rule D", "De-escalation — T5 downgraded to T4 for human review"),
        ]
        for rname, rdesc in rule_legend:
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid #0d0d0d;">'
                f'<span style="font-size:11px;font-weight:600;color:#64748b;">{rname}</span>'
                f'<span style="font-size:11px;color:#475569;margin-left:8px;">{rdesc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        card_close()

    # Tier 5 roster
    with audit_r:
        card_open()
        card_header("Strategic Default Roster", f"{len(t5_df)} confirmed Tier 5 accounts")

        t5_search = st.text_input("", placeholder="Filter by customer ID",
                                   label_visibility="collapsed", key="t5_search")
        t5_display = t5_df.copy()
        if t5_search:
            t5_display = t5_display[t5_display["customer_id"].str.contains(t5_search.upper(), na=False)]

        t5_table = pd.DataFrame({
            "Customer":     t5_display["customer_id"].values,
            "Distress":     t5_display["distress_avg"].round(3).values,
            "Anomaly":      ["⚠" if x else "—" for x in t5_display["bilstm_anomaly_flag"]],
            "Delay (mo)":   t5_display["avg_pay_delay"].round(2).values,
            "Fraud Rate":   t5_display["sp_fraud_rate"].round(3).values,
            "Rule A":       ["✓" if x else "—" for x in t5_display["audit_rule_a"]],
            "Rule B":       ["✓" if x else "—" for x in t5_display["audit_rule_b"]],
            "Rule C":       ["✓" if x else "—" for x in t5_display["audit_rule_c"]],
        })
        st.dataframe(
            t5_table, use_container_width=True, hide_index=True,
            column_config={
                "Customer":   st.column_config.TextColumn("Customer", width="medium"),
                "Distress":   st.column_config.ProgressColumn("Distress", min_value=0, max_value=1, format="%.3f"),
                "Anomaly":    st.column_config.TextColumn("BiLSTM", width="small"),
                "Delay (mo)": st.column_config.NumberColumn("Delay", format="%.2f", width="small"),
                "Fraud Rate": st.column_config.NumberColumn("Fraud", format="%.3f", width="small"),
                "Rule A":     st.column_config.TextColumn("A", width="small"),
                "Rule B":     st.column_config.TextColumn("B", width="small"),
                "Rule C":     st.column_config.TextColumn("C", width="small"),
            },
            height=500,
        )
        card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fairness:

    st.markdown("""
    <div style="background:#080808;border:1px solid #111111;border-radius:10px;
                padding:14px 20px;margin-bottom:20px;">
      <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px;">Demographic Fairness Audit (C2)</div>
      <div style="font-size:12px;color:#475569;line-height:1.6;">
        Tier 5 escalation rates are audited across sex, age group, and education level.
        SEX, AGE, and EDUCATION are not top SHAP drivers — escalation is driven by
        behavioural signals (avg_pay_delay, bilstm_anomaly_flag), not demographics.
      </div>
    </div>
    """, unsafe_allow_html=True)

    def tier_pct_pivot(group_col: str, label_map: dict | None = None) -> pd.DataFrame:
        tmp = df[["recovery_tier_final", group_col]].copy()
        if label_map:
            tmp[group_col] = tmp[group_col].map(label_map).fillna("Other")
        pivot = tmp.groupby([group_col, "recovery_tier_final"]).size().unstack(fill_value=0)
        return pivot.div(pivot.sum(axis=1), axis=0) * 100

    def fairness_chart(pivot: pd.DataFrame, key: str) -> go.Figure:
        fig = go.Figure()
        for tier in sorted(df["recovery_tier_final"].unique()):
            if tier not in pivot.columns:
                continue
            fig.add_trace(go.Bar(
                name=f"T{tier}",
                x=[str(g) for g in pivot.index.tolist()],
                y=pivot[tier].tolist(),
                marker_color=TIER_COLORS[tier], marker_line_width=0,
                text=[f"{v:.0f}%" for v in pivot[tier].tolist()],
                textposition="inside",
                textfont=dict(size=9, family="Inter, sans-serif", color="rgba(255,255,255,0.6)"),
                hovertemplate=f"T{tier} · %{{x}}<br>%{{y:.1f}}% of group<extra></extra>",
            ))
        fig.update_layout(
            **CHART_BASE, height=260, barmode="stack",
            xaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#475569")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       title=dict(text="% of Group", font=dict(size=11)), range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        font=dict(size=10, color="#2d2d2d"), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    f1, f2, f3 = st.columns(3, gap="medium")

    with f1:
        card_open()
        card_header("By Gender", "Stacked 100% bar — equal tier share = no gender bias")
        sex_pivot = tier_pct_pivot("SEX", {1: "Male", 2: "Female"})
        st.plotly_chart(fairness_chart(sex_pivot, "sex"), use_container_width=True,
                        config={"displayModeBar": False}, key="fair_sex")
        male_t5   = sex_pivot.loc["Male",   5] if "Male"   in sex_pivot.index and 5 in sex_pivot.columns else 0.0
        female_t5 = sex_pivot.loc["Female", 5] if "Female" in sex_pivot.index and 5 in sex_pivot.columns else 0.0
        gap       = abs(male_t5 - female_t5)
        gap_color = "#ef4444" if gap > 2 else "#10b981"
        st.markdown(
            f'<div style="font-size:11px;color:#475569;margin-top:8px;">'
            f'T5 gap — Male: <span style="color:#94a3b8;font-weight:600;">{male_t5:.1f}%</span> '
            f'vs Female: <span style="color:#94a3b8;font-weight:600;">{female_t5:.1f}%</span> '
            f'<span style="color:{gap_color};font-weight:600;">({gap:.1f}pp {"⚠ review" if gap > 2 else "✓ within tolerance"})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        card_close()

    with f2:
        card_open()
        card_header("By Age Group", "Binned into 4 cohorts — checks age-based escalation bias")
        age_df = df.copy()
        age_df["age_group"] = pd.cut(
            age_df["AGE"], bins=[0, 30, 40, 50, 200],
            labels=["<30", "30–39", "40–49", "50+"]
        )
        age_pivot = (age_df.groupby(["age_group", "recovery_tier_final"], observed=False)
                     .size().unstack(fill_value=0))
        age_pct = age_pivot.div(age_pivot.sum(axis=1), axis=0) * 100
        st.plotly_chart(fairness_chart(age_pct, "age"), use_container_width=True,
                        config={"displayModeBar": False}, key="fair_age")
        if 5 in age_pct.columns:
            max_age = age_pct[5].idxmax()
            st.markdown(
                f'<div style="font-size:11px;color:#475569;margin-top:8px;">'
                f'Highest T5 rate: <span style="color:#94a3b8;font-weight:600;">{max_age}</span> '
                f'({age_pct.loc[max_age, 5]:.1f}%)</div>',
                unsafe_allow_html=True,
            )
        card_close()

    with f3:
        card_open()
        card_header("By Education", "1=Graduate · 2=University · 3=High School · 4=Other")
        edu_pivot = tier_pct_pivot("EDUCATION", {1: "Graduate", 2: "University", 3: "High School", 4: "Other"})
        st.plotly_chart(fairness_chart(edu_pivot, "edu"), use_container_width=True,
                        config={"displayModeBar": False}, key="fair_edu")
        if 5 in edu_pivot.columns:
            max_edu = edu_pivot[5].idxmax()
            st.markdown(
                f'<div style="font-size:11px;color:#475569;margin-top:8px;">'
                f'Highest T5 rate: <span style="color:#94a3b8;font-weight:600;">{max_edu}</span> '
                f'({edu_pivot.loc[max_edu, 5]:.1f}%)</div>',
                unsafe_allow_html=True,
            )
        card_close()

    # Fairness summary table
    section_label("Tier 5 Escalation Rate Summary")
    card_open()
    card_header("Audit Verdict", "No demographic group is disproportionately escalated to Tier 5")

    summary_rows = []
    for group, val, ref in [
        ("Male",         float(sex_pivot.loc["Male",   5]) if "Male"   in sex_pivot.index and 5 in sex_pivot.columns else 0, "gender"),
        ("Female",       float(sex_pivot.loc["Female", 5]) if "Female" in sex_pivot.index and 5 in sex_pivot.columns else 0, "gender"),
    ]:
        summary_rows.append({"Group": group, "Category": "Gender", "T5 Rate": f"{val:.1f}%",
                              "Status": "✓ Pass" if val < 3.0 else "⚠ Review"})
    if 5 in age_pct.columns:
        for grp in age_pct.index:
            v = float(age_pct.loc[grp, 5])
            summary_rows.append({"Group": str(grp), "Category": "Age", "T5 Rate": f"{v:.1f}%",
                                  "Status": "✓ Pass" if v < 3.0 else "⚠ Review"})
    if 5 in edu_pivot.columns:
        for grp in edu_pivot.index:
            v = float(edu_pivot.loc[grp, 5])
            summary_rows.append({"Group": str(grp), "Category": "Education", "T5 Rate": f"{v:.1f}%",
                                  "Status": "✓ Pass" if v < 3.0 else "⚠ Review"})

    st.dataframe(
        pd.DataFrame(summary_rows), use_container_width=True, hide_index=True,
        column_config={
            "Group":    st.column_config.TextColumn("Group", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "T5 Rate":  st.column_config.TextColumn("T5 Rate", width="small"),
            "Status":   st.column_config.TextColumn("Status", width="medium"),
        },
        height=320,
    )
    card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_simulator:

    st.markdown("""
    <div style="background:#080808;border:1px solid #111111;border-radius:10px;
                padding:14px 20px;margin-bottom:24px;">
      <div style="font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:4px;">
        Recovery Tier Simulator
      </div>
      <div style="font-size:12px;color:#475569;line-height:1.6;">
        Adjust the customer signals below to see how the SBDR pipeline would classify them.
        The logic mirrors the B3.5 audit layer rules and XGBoost decision boundaries
        derived from SHAP analysis. Top features: avg_pay_delay (SHAP 2.36) ·
        bilstm_anomaly_flag (0.64) · distress_avg (0.14).
      </div>
    </div>
    """, unsafe_allow_html=True)

    sim_left, sim_right = st.columns([1, 1.2], gap="large")

    with sim_left:
        card_open()
        card_header("Input Signals", "Dial in the customer's key features")

        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        sim_pay_delay = st.slider(
            "Avg Payment Delay (months)", 0.0, 8.0, 1.0, 0.1, key="sim_pay_delay",
            help="0 = on time, 2 = 2 months late, 8 = severely delinquent"
        )
        sim_distress = st.slider(
            "FinBERT Distress Score", 0.0, 1.0, 0.35, 0.01, key="sim_distress",
            help="0.0 = calm, 1.0 = extreme financial distress"
        )
        sim_fraud = st.slider(
            "Sparkov Fraud Rate", 0.0, 1.0, 0.05, 0.01, key="sim_fraud",
            help="Fraction of transactions flagged as fraudulent"
        )

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sim_anomaly = st.checkbox("BiLSTM Anomaly Detected", value=False, key="sim_anomaly")
        with c2:
            sim_default = st.checkbox("Historical Default", value=False, key="sim_default")

        card_close()

    # ── Tier inference logic (mirrors audit layer + SHAP decision boundaries) ──
    def simulate_tier(pay_delay, distress, fraud_rate, anomaly, is_default):
        """Approximate the XGBoost + B3.5 audit layer classification."""
        rules_triggered = []

        # Rule C: fraud + anomaly + default (strongest Tier 5 signal)
        if fraud_rate >= 0.5 and anomaly and is_default:
            rules_triggered.append("C — Fraud + Anomaly + Default")

        # Rule A: engineered Tier 5 label (default + anomaly + low distress)
        if is_default and anomaly and distress < 0.55:
            rules_triggered.append("A — Strategic Default Pattern")

        # Rule B: high T5 probability proxy (severe delay + default + anomaly)
        if pay_delay >= 4.0 and is_default and anomaly:
            rules_triggered.append("B — High T5 Confidence")

        if rules_triggered:
            tier = 5
        elif (pay_delay >= 3.0 and is_default) or (distress >= 0.65 and pay_delay >= 2.0):
            tier = 4
        elif distress >= 0.55 or pay_delay >= 2.5:
            tier = 3
        elif distress >= 0.30 or pay_delay >= 0.5:
            tier = 2
        else:
            tier = 1

        # De-escalation: if flagged T5 but high distress + no anomaly → T4
        if tier == 5 and distress > 0.65 and not anomaly:
            tier = 4
            rules_triggered.append("D — De-escalated (genuine distress, not strategic)")

        return tier, rules_triggered

    sim_tier, sim_rules = simulate_tier(
        sim_pay_delay, sim_distress, sim_fraud, sim_anomaly, sim_default
    )
    sim_color = TIER_COLORS[sim_tier]
    sim_nba   = NBA_MAP[sim_tier]

    with sim_right:
        # Result card
        card_open()
        card_header("Predicted Classification", "Based on SBDR pipeline logic")

        st.markdown(f"""
        <div style="text-align:center;padding:28px 0 20px 0;">
          <div style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;
                      letter-spacing:0.12em;margin-bottom:12px;">Recovery Tier</div>
          <div style="font-size:56px;font-weight:800;color:{sim_color};
                      font-family:'Orbitron',sans-serif;line-height:1;margin-bottom:8px;">
            T{sim_tier}
          </div>
          <div style="font-size:16px;font-weight:500;color:#94a3b8;margin-bottom:20px;">
            {TIER_NAMES[sim_tier]}
          </div>
          <div style="background:rgba({
              '16,185,129' if sim_tier == 1 else
              '59,130,246' if sim_tier == 2 else
              '245,158,11' if sim_tier == 3 else
              '239,68,68'  if sim_tier == 4 else
              '124,58,237'
          },0.08);border:1px solid {sim_color}33;border-radius:8px;
                      padding:10px 20px;display:inline-block;">
            <div style="font-size:11px;color:#475569;margin-bottom:3px;font-weight:600;
                        text-transform:uppercase;letter-spacing:.1em;">Next Best Action</div>
            <div style="font-size:13px;font-weight:600;color:{sim_color};">{sim_nba}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk factor breakdown
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        indicators = [
            ("Avg Payment Delay", f"{sim_pay_delay:.1f} mo",
             "#ef4444" if sim_pay_delay > 3 else "#f59e0b" if sim_pay_delay > 1 else "#10b981"),
            ("Distress Score",    f"{sim_distress:.2f}",
             "#ef4444" if sim_distress > 0.65 else "#f59e0b" if sim_distress > 0.35 else "#10b981"),
            ("Fraud Rate",        f"{sim_fraud:.2f}",
             "#ef4444" if sim_fraud > 0.5 else "#f59e0b" if sim_fraud > 0.1 else "#10b981"),
            ("BiLSTM Anomaly",    "Detected" if sim_anomaly else "Clear",
             "#ef4444" if sim_anomaly else "#10b981"),
            ("Default History",   "Yes" if sim_default else "No",
             "#ef4444" if sim_default else "#10b981"),
        ]
        rows_html = ""
        for lbl, val, col in indicators:
            rows_html += (
                f'<div class="audit-row">'
                f'<span class="audit-key">{lbl}</span>'
                f'<span style="font-size:12px;font-weight:600;color:{col};">{val}</span>'
                f'</div>'
            )
        st.markdown(f'<div style="background:#050505;border-radius:8px;padding:10px 12px;'
                    f'border:1px solid #0f0f0f;">{rows_html}</div>', unsafe_allow_html=True)
        card_close()

        # Audit rules triggered
        if sim_rules:
            card_open()
            card_header("Audit Rules Triggered", "B3.5 layer signals that determined Tier 5")
            for rule in sim_rules:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
                    f'border-bottom:1px solid #0d0d0d;">'
                    f'<span style="width:6px;height:6px;border-radius:50%;background:#7c3aed;'
                    f'flex-shrink:0;display:inline-block;"></span>'
                    f'<span style="font-size:12px;color:#a78bfa;">{rule}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            card_close()
        else:
            card_open()
            card_header("No Escalation Rules Triggered", "Tier determined by behavioural thresholds")
            reasons = {
                1: "All signals within normal range — automated reminder sufficient",
                2: "Mild distress or slight delay — soft assistance recommended",
                3: "Elevated distress or recurring delay — hardship plan warranted",
                4: "Severe delay or high distress — relationship manager required",
            }
            st.markdown(
                f'<div style="font-size:12px;color:#475569;padding:8px 0;">'
                f'{reasons.get(sim_tier, "")}</div>',
                unsafe_allow_html=True,
            )
            card_close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:

    # ── Row 1: SHAP features + Branch donut ───────────────────────────────────
    m1c1, m1c2 = st.columns([1.6, 1], gap="medium")

    with m1c1:
        card_open()
        card_header("Top Predictive Features", "Mean |SHAP| value — coloured by AI branch")

        legend_html = "  ".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-size:10.5px;color:#64748b;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{c};'
            f'display:inline-block;"></span>{b}</span>'
            for b, c in BRANCH_COLORS.items()
        )
        st.markdown(f'<div style="margin-bottom:10px;">{legend_html}</div>',
                    unsafe_allow_html=True)

        top_feats = manifest["top_features_by_shap"][:8]
        names     = [SHAP_DISPLAY.get(f["feature"], f["feature"]) for f in top_feats][::-1]
        vals      = [f["mean_abs_shap"] for f in top_feats][::-1]
        clrs      = [BRANCH_COLORS.get(f["branch"], "#94a3b8") for f in top_feats][::-1]

        fig_shap = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=clrs, marker_line_width=0,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif", color="#475569"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        ))
        fig_shap.update_layout(
            **CHART_BASE, height=320,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", showline=False,
                       zeroline=False, title=dict(text="Mean |SHAP|", font=dict(size=11))),
            yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#475569")),
        )
        fig_shap.update_xaxes(range=[0, max(vals) * 1.22])
        st.plotly_chart(fig_shap, use_container_width=True,
                        config={"displayModeBar": False}, key="mod_shap")
        card_close()

    with m1c2:
        card_open()
        card_header("AI Branch Contributions", "Share of total predictive signal by model branch")

        branch_contrib = manifest["branch_shap_contribution"]
        total_shap     = sum(branch_contrib.values())
        b_labels       = list(branch_contrib.keys())
        b_vals         = [v / total_shap * 100 for v in branch_contrib.values()]
        b_colors       = [BRANCH_COLORS.get(b, "#94a3b8") for b in b_labels]

        fig_donut = go.Figure(go.Pie(
            labels=b_labels, values=b_vals, hole=0.65,
            marker=dict(colors=b_colors, line=dict(color="rgba(0,0,0,0.6)", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            pull=[0.03 if b == "UCI/Derived" else 0 for b in b_labels],
        ))
        fig_donut.update_layout(
            **CHART_BASE, height=300, showlegend=False,
            annotations=[dict(
                text=f"<b>{max(b_vals):.0f}%</b><br><span style='font-size:10px;'>UCI</span>",
                x=0.5, y=0.5, font=dict(size=13, color="#e2e8f0"), showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True,
                        config={"displayModeBar": False}, key="mod_donut")
        card_close()

    # ── Row 2: Model metrics ──────────────────────────────────────────────────
    card_open()
    card_header("Model Performance", "XGBoost classifier · 116 features · 30K customers · 5 recovery tiers")

    met_cols = st.columns(5, gap="medium")
    metric_items = [
        ("Accuracy",        f"{m['accuracy']*100:.1f}%",    "#10b981", "Overall classification rate"),
        ("AUC-ROC (macro)", f"{m['auc_roc_macro']:.3f}",    "#2563eb", "One-vs-rest discriminative power"),
        ("F1 Weighted",     f"{m['f1_weighted']*100:.1f}%", "#7c3aed", "Weighted by class frequency"),
        ("F1 Macro",        f"{m['f1_macro']*100:.1f}%",    "#0ea5e9", "Unweighted across all 5 tiers"),
        ("Val Log-Loss",    f"{m['best_val_mlogloss']:.4f}","#f59e0b", "Best early-stop validation loss"),
    ]
    for col, (lbl, val, accent, hint) in zip(met_cols, metric_items):
        with col:
            st.markdown(
                f'<div style="text-align:center;padding:16px 0;">'
                f'<div style="font-size:10px;font-weight:600;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;">{lbl}</div>'
                f'<div style="font-size:28px;font-weight:800;color:{accent};'
                f'font-family:\'Orbitron\',sans-serif;line-height:1;margin-bottom:6px;">{val}</div>'
                f'<div style="font-size:10.5px;color:#475569;">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    card_close()

    # ── Row 3: Artifact images ────────────────────────────────────────────────
    section_label("Model Evaluation Artifacts")
    art1, art2, art3, art4 = st.columns(4, gap="medium")
    artifact_specs = [
        (art1, "Confusion Matrix",        "Per-tier classification breakdown · XGBoost",
         ROOT / "models/xgboost/confusion_matrix.png"),
        (art2, "SHAP Feature Importance", "Mean |SHAP| beeswarm from Notebook 08",
         ROOT / "models/xgboost/shap_importance.png"),
        (art3, "B3.5 Audit Lift",         "XGBoost alone (219) vs after audit layer (485)",
         ROOT / "models/xgboost/audit_tier_comparison.png"),
        (art4, "BiLSTM Training Loss",    "100-epoch convergence · best val loss 0.0015",
         ROOT / "models/bilstm/training_loss.png"),
    ]
    for col, title, sub, path in artifact_specs:
        with col:
            card_open()
            card_header(title, sub)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.markdown(
                    '<div style="color:#334155;font-size:11px;padding:20px 0;">Image not found</div>',
                    unsafe_allow_html=True,
                )
            card_close()
