"""
RE-TCP Prototype — Aspect-Based, Requirement-Driven Framework for
Test Case Prioritization in Component-Based Systems.

Sidebar-dashboard layout (merged from the standalone HTML mockup).

Run with:  streamlit run app.py
"""

import base64
import hashlib
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from funnel_component import render_funnel_diagram

st.set_page_config(page_title="RE-TCP Prototype", layout="wide", page_icon="🧩")

BRAND = ["#BD582C", "#E48312", "#13294B", "#6B8E23", "#8A4D0A"]
NAVY = "#13294B"
RUST = "#BD582C"
ORANGE = "#E48312"

NFR_KEYWORDS = [
    "response time", "encrypt", "uptime", "concurrent", "accessible",
    "performance", "security", "seconds", "scalab", "load", "lag",
    "network", "error",
]

DATASETS = {
    "DS1 \u2014 PROMISE (Public Benchmark)": {
        "key": "ds1", "has_tc": True,
        "req_csv": "data/ds1_requirements.csv", "tc_csv": "data/ds1_test_cases.csv",
        "source": "PROMISE NFR Repository", "origin": "Public",
        "note": "625 requirements; 10 LLM-generated test cases (stratified RFR subset)",
    },
    "DS2 \u2014 RETRO.NET (Public Benchmark)": {
        "key": "ds2", "has_tc": False,
        "req_csv": "data/ds2_requirements.csv", "tc_csv": None,
        "source": "RETRO.NET Requirements Traceability Tool", "origin": "Public",
        "note": "14 requirements (stratified sample) \u2014 classification & prioritization only, no paired test cases",
    },
    "DS3 \u2014 EPCS Integration (Industrial, real)": {
        "key": "ds3", "has_tc": True,
        "req_csv": "data/ds3_requirements.csv", "tc_csv": "data/ds3_test_cases.csv",
        "source": "Drummond Group EPCS audit procedure", "origin": "Industrial (anonymized)",
        "note": "12 requirements, real paired test cases",
    },
    "DS4 \u2014 Manifests Web (Industrial, significant result)": {
        "key": "ds4", "has_tc": True,
        "req_csv": "data/ds4_requirements.csv", "tc_csv": "data/ds4_test_cases.csv",
        "source": "Internal product specification", "origin": "Industrial (anonymized)",
        "note": "14 requirements \u2014 APFD improvement statistically significant (p=0.029)",
    },
}

# Reported (manuscript) figures — not live-computed; used only on the Dashboard page
REPORTED_APFD_SINGLE_RUN = {"DS1": 0.910, "DS3": 0.940, "DS4": 0.958}
REPORTED_STAT_COMPARISON = [
    {"Dataset": "DS1 (n=10)", "KS-TCP (mean)": 0.503, "Random (mean)": 0.491, "Improvement": "+0.011", "p-value": 0.584, "Significant?": "No"},
    {"Dataset": "DS3 (n=12)", "KS-TCP (mean)": 0.514, "Random (mean)": 0.482, "Improvement": "+0.032", "p-value": 0.145, "Significant?": "No"},
    {"Dataset": "DS4 (n=14)", "KS-TCP (mean)": 0.533, "Random (mean)": 0.474, "Improvement": "+0.059", "p-value": 0.029, "Significant?": "Yes"},
]

PAGES = [
    "Dashboard",
    "Phase 1 \u2014 Requirement Prioritization",
    "Phase 2 \u2014 Test Case Selection & Prioritization",
    "Evaluation \u2014 APFD",
]


# --------------------------------------------------------------------------
# Theme: dark sidebar, KPI cards, badges, breadcrumb bar
# --------------------------------------------------------------------------
def inject_theme(active_page_index: int):
    st.markdown(f"""
    <style>
        html, body, [class*="css"]  {{ font-size: 20px; font-weight: 600; }}
        p, li, span, div {{ font-size: 1.08rem; font-weight: 600; }}
        h1 {{ font-size: 2.9rem !important; font-weight: 900 !important; }}
        h2 {{ font-size: 2.2rem !important; font-weight: 900 !important; }}
        h3 {{ font-size: 1.55rem !important; font-weight: 800 !important; }}
        label, .stSlider label, .stSelectbox label, .stMultiSelect label {{
            font-size: 1.05rem !important; font-weight: 700 !important;
        }}
        div[data-testid="stMetricLabel"] {{ font-size: 1rem !important; font-weight: 800 !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 2.1rem !important; font-weight: 900 !important; }}
        div[data-testid="stDataFrame"] * {{ font-weight: 600 !important; }}
        .stCaption, div[data-testid="stCaptionContainer"] {{ font-size: 0.95rem !important; font-weight: 600 !important; }}

        /* ---- Sidebar: dark navy dashboard nav ---- */
        section[data-testid="stSidebar"] {{
            background-color: {NAVY} !important;
            padding-top: 0 !important;
        }}
        section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15) !important; }}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
            color: #ffffff !important;
        }}

        section[data-testid="stSidebar"] .stButton button {{
            background-color: transparent !important;
            color: #cbd6e2 !important;
            text-align: left !important;
            font-weight: 800 !important;
            font-size: 1.08rem !important;
            border: none !important;
            border-left: 4px solid transparent !important;
            border-radius: 0 !important;
            padding: 12px 14px !important;
            width: 100%;
        }}
        section[data-testid="stSidebar"] .stButton button:hover {{
            background-color: rgba(255,255,255,0.08) !important;
            color: #ffffff !important;
        }}
        /* Active nav item — targets the Nth button in the sidebar (1-indexed) */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:nth-child({active_page_index}) .stButton button {{
            background-color: {ORANGE} !important;
            color: #ffffff !important;
            border-left: 4px solid #ffffff !important;
        }}

        section[data-testid="stSidebar"] .stSelectbox label {{
            color: #9fb4c7 !important; font-size: 0.78rem !important;
            text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700 !important;
        }}
        .nav-desc {{
            color: #b8c8d8; font-size: 0.85rem; font-weight: 600; padding: 0 14px 12px 14px; margin-top: -4px; line-height: 1.35;
        }}
        .dataset-label {{
            color: #E48312; font-size: 0.95rem; font-weight: 800; padding: 4px 0 8px 0;
        }}

        /* ---- Breadcrumb / top bar ---- */
        .re-topbar {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 4px 18px 4px; border-bottom: 1px solid #e6e6e6; margin-bottom: 22px;
        }}
        .re-breadcrumb {{ font-size: 1rem; color: #888; font-weight: 600; }}
        .re-breadcrumb b {{ color: {NAVY}; }}

        /* ---- KPI cards ---- */
        .kpi-card {{
            background: #ffffff; border-radius: 12px; padding: 16px 18px;
            border: 1px solid #eaeaea; border-left: 5px solid {RUST};
            box-shadow: 0 2px 6px rgba(0,0,0,0.05); height: 100%;
        }}
        .kpi-label {{ font-size: 0.78rem; color: #888; font-weight: 800; text-transform: uppercase; letter-spacing: 0.4px; }}
        .kpi-value {{ font-size: 2.3rem; font-weight: 900; color: {NAVY}; margin-top: 4px; }}
        .kpi-note {{ font-size: 0.82rem; color: #999; margin-top: 3px; font-weight: 600; }}

        /* ---- Selector cards (Phase 1 / Phase 2 / Dataset) ---- */
        .selector-card {{
            background: #ffffff; border-radius: 16px; padding: 22px 20px; text-align: center;
            border: 2px solid #eaeaea; box-shadow: 0 3px 10px rgba(0,0,0,0.06); height: 100%;
            transition: border-color 0.15s;
        }}
        .selector-icon {{ font-size: 2.6rem; margin-bottom: 8px; }}
        .selector-title {{ font-size: 1.15rem; font-weight: 900; color: {NAVY}; margin-bottom: 6px; }}
        .selector-desc {{ font-size: 0.88rem; color: #555; font-weight: 600; line-height: 1.4; }}

        /* ---- Badges ---- */
        .badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 0.72rem; font-weight: 700; }}
        .badge-sig {{ background: #D9EAD3; color: #274e13; }}
        .badge-ns {{ background: #f0f0f0; color: #888; }}

        /* ---- Section headers in main content ---- */
        h3 {{ color: {NAVY} !important; font-weight: 700 !important; border-left: 5px solid {ORANGE}; padding-left: 10px; }}
        div[data-testid="stMetricValue"] {{ color: {RUST} !important; font-weight: 800 !important; }}
        .stButton button, .stDownloadButton button {{
            background-color: {RUST}; color: #fff; font-weight: 700; border-radius: 8px; border: none;
        }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, note=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def breadcrumb(page_name):
    st.markdown(f"""
    <div class="re-topbar">
        <div class="re-breadcrumb">Pages / <b>{page_name}</b></div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*candidates):
    """Return the first existing path, so the app works with or without data/ and assets/ folders."""
    for c in candidates:
        for p in (c, os.path.join(APP_DIR, c)):
            if os.path.exists(p):
                return p
    return candidates[-1]


@st.cache_data
def load_csv(path):
    return pd.read_csv(find_file(path, os.path.basename(path)))


def classify_fr_nfr(text: str) -> str:
    t = text.lower()
    return "NFR" if any(k in t for k in NFR_KEYWORDS) else "FR"


def normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0
    return (series - lo) / (hi - lo)


def k_medoids(X, k, max_iter=50, seed=42):
    """Minimal K-Medoids (PAM-style) implementation, dependency-free."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    medoid_idx = rng.choice(n, k, replace=False)
    for _ in range(max_iter):
        D = pairwise_distances(X, X[medoid_idx])
        labels = D.argmin(axis=1)
        new_medoids = medoid_idx.copy()
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                continue
            sub_D = pairwise_distances(X[members], X[members])
            costs = sub_D.sum(axis=1)
            new_medoids[ci] = members[costs.argmin()]
        if np.array_equal(new_medoids, medoid_idx):
            break
        medoid_idx = new_medoids
    D = pairwise_distances(X, X[medoid_idx])
    labels = D.argmin(axis=1)
    return labels


def apfd(order: list, fault_ids: list) -> float:
    n = len(order)
    m = len([f for f in fault_ids if f in order])
    if n == 0 or m == 0:
        return 0.0
    tf_sum = sum(order.index(f) + 1 for f in fault_ids if f in order)
    return 1 - (tf_sum / (n * m)) + 1 / (2 * n)


# --------------------------------------------------------------------------
# Upload-your-own-dataset support: parsing, aspect classification,
# default priority inputs and requirement-test alignment
# --------------------------------------------------------------------------
UPLOAD_OPTION = "\U0001F4E4 Upload your own dataset"

# Aspect lexicon (ISO/IEC 25010-style quality characteristics). Order matters:
# the first aspect whose keywords match wins.
ASPECT_KEYWORDS = {
    "Security": ["secur", "encrypt", "password", "authenticat", "authoriz", "login", "log in", "access control",
                 "permission", "privacy", "confidential", "audit", "two-factor", "2fa", "credential", "role",
                 "attack", "malicious", "tamper", "unauthori", "sensitive", "ssl", "tls", "token"],
    "Reliability": ["reliab", "availability", "be available", "uptime", " fault", "failure", "recover", "backup", "crash", "downtime",
                    "fail-safe", "failover", "robust", "error handling", "24/7", "99."],
    "Performance": ["performance", "response time", "seconds", "millisecond", "latency", "throughput",
                    "fast", "speed", " load ", "load time", "concurrent", "simultaneous", " lag"],
    "Scalability": ["scalab", "scale", "increase in users", "growth", "expand"],
    "Usability": ["usab", "user-friendly", "user friendly", "easy to", "intuitive", "learn", "accessib",
                  "look and feel", "appearance", "colour", "font size", "readable", "user interface shall"],
    "Compatibility": ["compatib", "browser", "platform", "operating system", "interoperab", "integrate with",
                      "mobile device", "portab"],
    "Maintainability": ["maintainab", "modular", "extensib"],
    "Legal/Compliance": ["legal", "complian", "regulat", " law", "hipaa", "gdpr"],
}
# Criticality of each aspect for fault-risk weighting (1 = low, 5 = high)
ASPECT_CRITICALITY = {
    "Security": 5, "Reliability": 5, "Performance": 4, "Legal/Compliance": 4, "Scalability": 3,
    "Compatibility": 3, "Usability": 3, "Maintainability": 2, "Functional": 3,
}
CRITICAL_ASPECTS = {"Security", "Reliability", "Performance", "Legal/Compliance"}

TEXT_COL_CANDIDATES = ["requirement_text", "requirement", "requirements", "text", "req_text", "description", "statement"]
ID_COL_CANDIDATES = ["req_id", "id", "requirement_id", "reqid", "no", "number"]
TC_TEXT_CANDIDATES = ["description", "test_case", "testcase", "test case", "test_description", "steps", "title", "name", "text"]
TC_ID_CANDIDATES = ["tc_id", "test_id", "testcase_id", "test_case_id", "id"]
TC_LINK_CANDIDATES = ["linked_req_id", "req_id", "requirement_id", "linked_requirement", "requirement"]

SAMPLE_REQ_CSV = (
    "req_id,requirement_text\n"
    "REQ-01,The system shall allow a registered user to log in with a username and password.\n"
    "REQ-02,All passwords shall be stored encrypted and never displayed in plain text.\n"
    "REQ-03,The system shall let the user search products by name and category.\n"
    "REQ-04,Search results shall be returned within 2 seconds for 500 concurrent users.\n"
    "REQ-05,The system shall allow the user to add products to a shopping cart.\n"
    "REQ-06,The checkout page shall be easy to use and readable on mobile devices.\n"
    "REQ-07,The system shall be available 99.9% of the time and recover from a crash within 5 minutes.\n"
    "REQ-08,The system shall email an order confirmation after successful payment.\n"
)
SAMPLE_TC_CSV = (
    "tc_id,description\n"
    "TC-01,Verify user can log in with a valid username and password\n"
    "TC-02,Verify login is rejected with an invalid password\n"
    "TC-03,Verify stored passwords are encrypted in the database\n"
    "TC-04,Verify product search by name returns matching products\n"
    "TC-05,Verify search response time stays under 2 seconds with 500 concurrent users\n"
    "TC-06,Verify product can be added to the shopping cart\n"
    "TC-07,Verify checkout page layout is readable on a mobile device\n"
    "TC-08,Verify system recovers within 5 minutes after a simulated crash\n"
    "TC-09,Verify order confirmation email is sent after payment\n"
    "TC-10,Verify searching by category returns products of that category\n"
)


def classify_aspect(text: str) -> str:
    t = " " + str(text).lower() + " "
    for aspect, kws in ASPECT_KEYWORDS.items():
        if any(k in t for k in kws):
            return aspect
    return "Functional"


def _pick_col(df, candidates):
    lower = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def read_uploaded_table(uploaded):
    """Read an uploaded CSV / Excel / TXT file into a DataFrame."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    import io
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    if name.endswith(".txt"):
        lines = [ln.strip() for ln in data.decode("utf-8", errors="ignore").splitlines() if ln.strip()]
        return pd.DataFrame({"requirement_text": lines})
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


def prepare_uploaded_requirements(df):
    """Normalise an uploaded requirement list to the columns the pipeline expects."""
    df = df.dropna(how="all").copy()
    text_col = _pick_col(df, TEXT_COL_CANDIDATES)
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError("Could not find a column containing requirement text.")
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    id_col = _pick_col(df, ID_COL_CANDIDATES)
    out = pd.DataFrame()
    out["requirement_text"] = df[text_col].astype(str).str.strip()
    out = out[out["requirement_text"].str.len() > 0]
    if looks_like_package_list(out["requirement_text"].tolist()):
        raise ValueError("this looks like a Python package list (e.g. the app's own requirements.txt), "
                         "not a list of software requirements such as 'The system shall allow users to log in'")
    if id_col is not None and id_col != text_col:
        out["req_id"] = df.loc[out.index, id_col].astype(str).str.strip()
    else:
        out["req_id"] = [f"REQ-{i+1:02d}" for i in range(len(out))]
    out = out.reset_index(drop=True)
    out["aspect"] = out["requirement_text"].apply(classify_aspect)
    out["fr_nfr_given"] = np.where(out["aspect"] == "Functional", "FR", "NFR")
    must_words = ("must", "shall", "critical", "mandatory", "required")
    for col, default in [("stakeholder_priority", None), ("change_frequency", 1), ("historical_defects", 0)]:
        src = _pick_col(df, [col, col.replace("_", " "), col.split("_")[0]])
        if src is not None:
            vals = pd.to_numeric(df[src], errors="coerce").reset_index(drop=True)
        else:
            vals = pd.Series([np.nan] * len(out))
        if col == "stakeholder_priority":
            # Aspect-based default: criticality of the quality aspect, +1 for mandatory wording (max 5)
            aspect_default = out.apply(
                lambda r: min(5, ASPECT_CRITICALITY.get(r["aspect"], 3)
                          + (1 if any(w in r["requirement_text"].lower() for w in must_words) else 0)), axis=1)
            out[col] = vals.fillna(aspect_default).astype(float)
        else:
            out[col] = vals.fillna(default).astype(float)
    return out[["req_id", "requirement_text", "aspect", "fr_nfr_given",
                "stakeholder_priority", "change_frequency", "historical_defects"]]


import re as _re
_PKG_LINE = _re.compile(r"^[A-Za-z0-9_.\-\[\]]+\s*(==|>=|<=|~=|!=|>|<)\s*[0-9][0-9A-Za-z.*]*\s*$")


def looks_like_package_list(texts):
    texts = [t for t in texts if t.strip()]
    return bool(texts) and sum(bool(_PKG_LINE.match(t.strip())) for t in texts) >= 0.6 * len(texts)


ASPECT_TEST_TEMPLATES = {
    "Functional": "Functional test \u2014 execute the main and alternate flows and verify: {r}",
    "Security": "Security test \u2014 attempt unauthorized/invalid access and verify: {r}",
    "Reliability": "Reliability test \u2014 simulate a failure and verify: {r}",
    "Performance": "Performance test \u2014 run under expected peak load and verify: {r}",
    "Scalability": "Scalability test \u2014 increase users/data volume and verify: {r}",
    "Usability": "Usability test \u2014 observe a representative user and verify: {r}",
    "Compatibility": "Compatibility test \u2014 repeat on each supported platform/browser and verify: {r}",
    "Maintainability": "Maintainability review \u2014 inspect the change impact and verify: {r}",
    "Legal/Compliance": "Compliance audit \u2014 check against the regulation and verify: {r}",
    "NFR (general)": "Quality test \u2014 verify: {r}",
}


def generate_test_cases(reqs):
    """Create one aspect-specific test case per requirement (used when no test case file is uploaded)."""
    rows = []
    for i, r in reqs.reset_index(drop=True).iterrows():
        text = str(r["requirement_text"]).strip().rstrip(".")
        tmpl = ASPECT_TEST_TEMPLATES.get(r.get("aspect", "Functional"), ASPECT_TEST_TEMPLATES["Functional"])
        rows.append({"tc_id": f"TC-{i+1:02d}", "description": tmpl.format(r=text),
                     "linked_req_id": r["req_id"], "alignment_score": 1.0,
                     "alignment_method": "Generated from requirement",
                     "redundancy_text": text})
    return pd.DataFrame(rows)


ALIGN_THRESHOLD = 0.05


def prepare_uploaded_tests(df, reqs):
    """Normalise uploaded test cases and align each one to a requirement.
    Uses the file's linked_req_id when present, otherwise TF-IDF cosine similarity."""
    df = df.dropna(how="all").copy()
    text_col = _pick_col(df, TC_TEXT_CANDIDATES)
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError("Could not find a column containing test case descriptions.")
        text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    id_col = _pick_col(df, TC_ID_CANDIDATES)
    link_col = _pick_col(df, TC_LINK_CANDIDATES)
    if link_col == text_col:
        link_col = None
    out = pd.DataFrame()
    out["description"] = df[text_col].astype(str).str.strip()
    out["tc_id"] = (df[id_col].astype(str).str.strip() if id_col is not None and id_col != text_col
                    else [f"TC-{i+1:02d}" for i in range(len(df))])
    out = out[out["description"].str.len() > 0].reset_index(drop=True)
    link_vals = (df[link_col].astype(str).str.strip().reset_index(drop=True)
                 if link_col is not None else pd.Series([""] * len(out)))

    vec = TfidfVectorizer(stop_words="english")
    vec.fit(pd.concat([reqs["requirement_text"], out["description"]]))
    sim = cosine_similarity(vec.transform(out["description"]), vec.transform(reqs["requirement_text"]))
    req_ids = reqs["req_id"].tolist()
    linked, scores, methods = [], [], []
    for i in range(len(out)):
        given = link_vals.iloc[i] if i < len(link_vals) else ""
        if given and given.lower() != "nan" and given in req_ids:
            j = req_ids.index(given)
            linked.append(given); scores.append(float(sim[i, j])); methods.append("Given in file")
        else:
            j = int(sim[i].argmax()) if len(req_ids) else 0
            if len(req_ids) and sim[i, j] >= ALIGN_THRESHOLD:
                linked.append(req_ids[j]); scores.append(float(sim[i, j])); methods.append("Auto (text similarity)")
            else:
                linked.append("UNALIGNED"); scores.append(float(sim[i, j]) if len(req_ids) else 0.0)
                methods.append("No matching requirement")
    out["linked_req_id"] = linked
    out["alignment_score"] = scores
    out["alignment_method"] = methods
    return out[["tc_id", "description", "linked_req_id", "alignment_score", "alignment_method"]]


# --------------------------------------------------------------------------
# Sidebar: logo, navigation, dataset selector
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

with st.sidebar:
    logo_path = find_file(os.path.join("assets", "prioritization_icon.png"), "prioritization_icon.png")
    lcol1, lcol2, lcol3 = st.columns([1, 2, 1])
    with lcol2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=160)
    st.markdown("<hr>", unsafe_allow_html=True)

    NAV_INFO = {
        "Dashboard": "\U0001F4CA Overview, KPIs & results across all datasets",
        "Phase 1 \u2014 Requirement Prioritization": "\U0001F4DD Classify & rank requirements (LDA + OurRank)",
        "Phase 2 \u2014 Test Case Selection & Prioritization": "\U0001F9EA Select & order test cases (K-Medoids)",
        "Evaluation \u2014 APFD": "\U0001F4C8 Fault-detection results (APFD)",
    }
    for p in PAGES:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
        st.markdown(f"<div class='nav-desc'>{NAV_INFO[p]}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='dataset-label'>\U0001F5C2\uFE0F DATASET \u2014 choose which case study to explore</div>",
                unsafe_allow_html=True)
    dataset_names = list(DATASETS.keys()) + [UPLOAD_OPTION]
    default_ds_index = next((i for i, n in enumerate(dataset_names) if DATASETS[n]["key"] == "ds3"), 0)
    dataset_name = st.selectbox("Select dataset", dataset_names, index=default_ds_index, label_visibility="collapsed")
    if dataset_name == UPLOAD_OPTION:
        st.markdown("<div class='nav-desc'>Upload your files in the panel at the top of the page \u2192</div>",
                    unsafe_allow_html=True)

active_index = 2 * PAGES.index(st.session_state.page) + 3  # buttons now alternate with description captions
inject_theme(active_index)

UPLOAD_MODE = dataset_name == UPLOAD_OPTION


@st.cache_data
def _parse_upload(name, data, kind):
    class _F:
        pass
    f = _F()
    f.name, f.getvalue = name, (lambda: data)
    return read_uploaded_table(f)


if UPLOAD_MODE:
    no_file_yet = st.session_state.get("up_req") is None
    with st.expander("\U0001F4E4 Your dataset \u2014 upload requirements and test cases", expanded=no_file_yet):
        st.markdown(
            "**1. Requirements file** (required): CSV, Excel or TXT. Needs one column with the requirement text "
            "(e.g. `requirement_text`). Optional columns: `req_id`, `stakeholder_priority` (1\u20135), "
            "`change_frequency`, `historical_defects`. A TXT file can simply list one requirement per line.\n\n"
            "**2. Test cases file** (optional): CSV or Excel with a `description` column "
            "and optionally `tc_id` and `linked_req_id`. If `linked_req_id` is missing, each test case is "
            "aligned to its closest requirement automatically. If you have no test cases, leave this empty and "
            "the app generates one aspect-specific test case per requirement.")
        d1, d2 = st.columns(2)
        d1.download_button("\u2B07 Sample requirements file", SAMPLE_REQ_CSV.encode(),
                           file_name="sample_requirements.csv", mime="text/csv", use_container_width=True)
        d2.download_button("\u2B07 Sample test cases file", SAMPLE_TC_CSV.encode(),
                           file_name="sample_test_cases.csv", mime="text/csv", use_container_width=True)
        u1, u2 = st.columns(2)
        req_file = u1.file_uploader("Requirements file", type=["csv", "xlsx", "xls", "txt"], key="up_req")
        tc_file = u2.file_uploader("Test cases file (optional)", type=["csv", "xlsx", "xls"], key="up_tc")
        auto_tc = st.checkbox("No test case file? Auto-generate one aspect-specific test case per requirement",
                              value=True, key="up_autotc")

    if req_file is None:
        breadcrumb("Upload your dataset")
        st.info("\U0001F4E4 Upload a **requirements file** above to start. Not sure of the format? Download the "
                "sample files, open them in Excel, replace the rows with your own requirements and test cases, "
                "save as CSV and upload.")
        st.stop()

    try:
        reqs_raw = prepare_uploaded_requirements(_parse_upload(req_file.name, req_file.getvalue(), "req"))
        if len(reqs_raw) < 2:
            raise ValueError("The file needs at least 2 requirements.")
        tcs_raw = (prepare_uploaded_tests(_parse_upload(tc_file.name, tc_file.getvalue(), "tc"), reqs_raw)
                   if tc_file is not None else None)
        if tcs_raw is not None and len(tcs_raw) == 0:
            tcs_raw = None
        tc_generated = tcs_raw is None and auto_tc
        if tc_generated:
            tcs_raw = generate_test_cases(reqs_raw)
    except Exception as e:
        st.error(f"Could not read your file: {e}. Please check it matches the sample format.")
        st.stop()

    has_tc = tcs_raw is not None
    sig = hashlib.md5(req_file.getvalue() + (tc_file.getvalue() if tc_file is not None else b"")
                      + (b"gen" if tc_generated else b"")).hexdigest()[:10]
    cfg = {"key": f"upload_{sig}", "has_tc": has_tc, "source": f"Your dataset ({req_file.name})",
           "origin": "User-uploaded",
           "note": f"{len(reqs_raw)} requirements" + (
               f"; {len(tcs_raw)} test cases" + (" (auto-generated)" if tc_generated else "")
               if has_tc else "; no test cases uploaded")}
    st.success(f"Loaded **{len(reqs_raw)} requirements** "
               + (f"and **{len(tcs_raw)} test cases**" + (" (auto-generated from your requirements)" if tc_generated else "")
                  if has_tc else "(no test case file \u2014 Phase 2 needs one)")
               + ". Go to **Phase 1** in the sidebar.")
else:
    cfg = DATASETS[dataset_name]
    reqs_raw = load_csv(cfg["req_csv"]).copy()
    has_tc = cfg["has_tc"]
    tcs_raw = load_csv(cfg["tc_csv"]).copy() if has_tc else None
LARGE = len(reqs_raw) > 60

page = st.session_state.page

# ==========================================================================
# DASHBOARD
# ==========================================================================
if page == "Dashboard":
    breadcrumb("Dashboard")
    logo_col, title_col = st.columns([1, 8])
    with logo_col:
        proc_logo_path = find_file(os.path.join("assets", "prioritization_icon.png"), "prioritization_icon.png")
        if os.path.exists(proc_logo_path):
            st.image(proc_logo_path, width=150)
    with title_col:
        st.markdown(f"<h2 style='color:{NAVY}; margin-bottom:0;'>Datasets Overview</h2>", unsafe_allow_html=True)
        st.caption("Aspect-Based, Requirement-Driven Framework \u2014 case-study evaluation across DS1 (PROMISE), "
                   "DS3 (EPCS Integration), and DS4 (Manifests Web).")

    st.markdown("<h3>Start Here \u2014 Choose What to Explore</h3>", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("""
        <div class="selector-card">
            <div class="selector-icon">\U0001F4DD</div>
            <div class="selector-title">Phase 1</div>
            <div class="selector-desc">Classify requirements as FR/NFR (LDA topic modeling) and rank them by
            priority (OurRank, 5-stage algorithm). <b>Start here</b> to see how raw requirements become an
            ordered priority list.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Phase 1 \u2192", key="goto_phase1", use_container_width=True):
            st.session_state.page = PAGES[1]
            st.rerun()
    with sc2:
        st.markdown("""
        <div class="selector-card">
            <div class="selector-icon">\U0001F9EA</div>
            <div class="selector-title">Phase 2</div>
            <div class="selector-desc">Map prioritized requirements to test cases, cluster them (K-Medoids),
            and produce the final execution order. <b>Use this</b> to see redundant/irrelevant test cases
            get filtered out automatically.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Phase 2 \u2192", key="goto_phase2", use_container_width=True):
            st.session_state.page = PAGES[2]
            st.rerun()
    with sc3:
        st.markdown("""
        <div class="selector-card">
            <div class="selector-icon">\U0001F5C2\uFE0F</div>
            <div class="selector-title">Dataset</div>
            <div class="selector-desc">Switch between DS1 (public benchmark, 625 reqs), DS2 (public benchmark,
            14 reqs), DS3 (real industrial EPCS data), and DS4 (real industrial data, statistically significant
            result). <b>Pick one</b> in the sidebar to drive every phase below.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; padding-top:10px; font-weight:800; color:{ORANGE};'>"
                    f"\u2190 Selector is in the sidebar</div>", unsafe_allow_html=True)

    st.markdown("<h3>The Full Pipeline \u2014 All 3 Phases</h3>", unsafe_allow_html=True)
    pipeline_img_path = find_file(os.path.join("assets", "combined_pipeline.png"), "combined_pipeline.png")
    if os.path.exists(pipeline_img_path):
        st.image(pipeline_img_path, use_container_width=True)

    total_reqs = sum(len(load_csv(d["req_csv"])) for d in DATASETS.values())
    total_tcs = sum(len(load_csv(d["tc_csv"])) for d in DATASETS.values() if d["has_tc"])
    best_apfd_ds, best_apfd_val = max(REPORTED_APFD_SINGLE_RUN.items(), key=lambda kv: kv[1])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Requirements", f"{total_reqs}", "625 + 12 + 14 across all datasets")
    with c2:
        kpi_card("Test Cases", f"{total_tcs}", "10 (DS1) + 12 (DS3) + 12 (DS4)")
    with c3:
        kpi_card("Best APFD (single-run)", f"{best_apfd_val:.3f}", f"{best_apfd_ds} \u00b7 reported value")
    with c4:
        kpi_card("Significant Result", "p = 0.029", "DS4 vs. Random Prioritization")

    st.write("")
    cL, cR = st.columns([1, 1.15])
    with cL:
        st.markdown("<h3>Dataset Summary</h3>", unsafe_allow_html=True)
        summary_rows = [
            {"Dataset": d["key"].upper(), "Source": d["source"], "Reqs": len(load_csv(d["req_csv"])), "Origin": d["origin"]}
            for name, d in DATASETS.items()
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.caption("A stratified sample was used for DS1 and DS2 to keep the classification/prioritization "
                   "walkthrough inspectable; DS3 and DS4 use their full real requirement sets.")
    with cR:
        st.markdown("<h3>Statistical Evaluation of Datasets</h3>", unsafe_allow_html=True)
        st.caption("Repeated-execution APFD (30 random-seed repetitions, paired t-test): proposed method vs. "
                   "random-order baseline, per dataset.")
        stat_long_rows = []
        for row in REPORTED_STAT_COMPARISON:
            ds = row["Dataset"]
            stat_long_rows.append({"Dataset": ds, "Method": "Proposed (KS-TCP)", "APFD": row["KS-TCP (mean)"]})
            stat_long_rows.append({"Dataset": ds, "Method": "Random baseline", "APFD": row["Random (mean)"]})
        stat_long_df = pd.DataFrame(stat_long_rows)
        fig_stat = px.bar(
            stat_long_df, x="Dataset", y="APFD", color="Method", barmode="group",
            color_discrete_map={"Proposed (KS-TCP)": BRAND[0], "Random baseline": "#b0b0b0"},
            text="APFD",
        )
        fig_stat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_stat.update_layout(height=380, yaxis_range=[0, 0.65], legend_title_text="")
        # annotate the one statistically significant result
        fig_stat.add_annotation(x="DS4 (n=14)", y=0.60, text="\u2605 significant (p = 0.029)",
                                 showarrow=False, font=dict(color=RUST, size=15, family="Arial Black"))
        st.plotly_chart(fig_stat, use_container_width=True)

    st.markdown("<h3>APFD by Dataset (illustrative single-run)</h3>", unsafe_allow_html=True)
    apfd_df = pd.DataFrame({"Dataset": list(REPORTED_APFD_SINGLE_RUN.keys()),
                             "APFD": list(REPORTED_APFD_SINGLE_RUN.values())})
    fig_dash = px.bar(apfd_df, x="Dataset", y="APFD", color="Dataset",
                       color_discrete_sequence=[BRAND[0], BRAND[1], BRAND[2]], text="APFD")
    fig_dash.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_dash.update_layout(showlegend=False, height=360, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_dash, use_container_width=True)

    st.markdown("<h3>How the Framework Works</h3>", unsafe_allow_html=True)
    st.markdown(render_funnel_diagram(), unsafe_allow_html=True)

# ==========================================================================
# PHASE 1
# ==========================================================================
elif page == "Phase 1 \u2014 Requirement Prioritization":
    breadcrumb(f"Phase 1 \u00b7 {cfg['source']}")
    st.info(f"**{cfg['source']}** \u00b7 {cfg['origin']} \u00b7 {cfg['note']}")

    st.subheader("1. Requirement Gathering")
    if LARGE:
        st.caption(f"Showing a preview of {len(reqs_raw)} total requirements (scroll for more).")
    st.dataframe(reqs_raw[["req_id", "requirement_text"]], use_container_width=True, hide_index=True, height=260)

    st.subheader("1a. Data Quality Check \u2014 Duplicate Requirements")
    dup_counts = reqs_raw["req_id"].value_counts()
    dup_ids = dup_counts[dup_counts > 1]
    n_total = len(reqs_raw)
    n_unique = reqs_raw["req_id"].nunique()

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Total rows loaded", n_total)
    qc2.metric("Unique req_id values", n_unique)
    qc3.metric("Duplicate req_id groups", len(dup_ids), delta=None if len(dup_ids) == 0 else f"{n_total - n_unique} extra rows",
               delta_color="inverse")

    if len(dup_ids) > 0:
        exact_dupe_groups, conflicting_groups = 0, 0
        for rid in dup_ids.index:
            group = reqs_raw[reqs_raw["req_id"] == rid]
            if group["requirement_text"].nunique() == 1:
                exact_dupe_groups += 1
            else:
                conflicting_groups += 1

        st.warning(f"**{len(dup_ids)} req_id value(s) appear more than once** \u2014 "
                   f"{exact_dupe_groups} are exact duplicates (identical text), "
                   f"{conflicting_groups} have the *same ID but different text* (a data integrity issue, not just duplication).")

        with st.expander(f"See the {len(dup_ids)} duplicated req_id(s)", expanded=False):
            dup_detail = reqs_raw[reqs_raw["req_id"].isin(dup_ids.index)].sort_values("req_id")
            st.dataframe(dup_detail[["req_id", "requirement_text"]], use_container_width=True, hide_index=True, height=240)

        keep_dupes = st.checkbox(
            "Keep duplicates and run the pipeline anyway (to see their effect on the framework)",
            value=False,
            help="Unchecked (default): duplicates are removed before Phase 1 runs, keeping the first occurrence "
                 "of each req_id. Checked: duplicates are kept, so you can directly observe how they distort "
                 "downstream results \u2014 useful for testing robustness, not recommended for a real evaluation run.",
        )

        if not keep_dupes:
            reqs_raw = reqs_raw.drop_duplicates(subset="req_id", keep="first").reset_index(drop=True)
            st.success(f"Deduplicated: proceeding with {len(reqs_raw)} unique requirements.")
        else:
            st.error(f"Running with duplicates present ({n_total} rows, only {n_unique} unique). Watch for: "
                     "the same requirement occupying two slots in the ourRank priority list, the cosine-similarity "
                     "heatmap and traceability matrix breaking on non-unique labels, and inflated-looking topic "
                     "counts. This is intentionally left broken when this box is checked, so you can see exactly "
                     "where and how duplicate IDs corrupt the pipeline.")
    else:
        st.success("No duplicate req_id values found \u2014 every requirement in this dataset is unique.")

    st.subheader("2. Requirement Classification (Topic Modeling)")
    n_topics = st.slider("Number of topics (K)", 2, 6, 3, key="n_topics")

    vec = TfidfVectorizer(stop_words="english", max_features=400)
    X = vec.fit_transform(reqs_raw["requirement_text"])

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
    topic_dist = lda.fit_transform(X)
    reqs = reqs_raw.copy()
    reqs["topic"] = topic_dist.argmax(axis=1)

    if UPLOAD_MODE:
        reqs["fr_nfr"] = reqs["fr_nfr_given"]
        st.caption("FR/NFR labels and quality aspects are assigned by the aspect-based classifier "
                   "(ISO/IEC 25010 quality characteristics: Security, Reliability, Performance, Usability, ...). "
                   "A requirement with no quality aspect is a Functional Requirement (FR).")
    elif "fr_nfr_given" in reqs.columns and reqs["fr_nfr_given"].notna().all():
        reqs["fr_nfr"] = reqs["fr_nfr_given"]
        st.caption("FR/NFR labels shown are the dataset's original ground-truth labels (not the heuristic classifier).")
    else:
        reqs["fr_nfr"] = reqs["requirement_text"].apply(classify_fr_nfr)

    if "aspect" not in reqs.columns:
        reqs["aspect"] = reqs["requirement_text"].apply(classify_aspect)
    reqs.loc[reqs["fr_nfr"] == "FR", "aspect"] = "Functional"
    reqs.loc[(reqs["fr_nfr"] == "NFR") & (reqs["aspect"] == "Functional"), "aspect"] = "NFR (general)"

    st.dataframe(
        reqs[["req_id", "requirement_text", "fr_nfr", "aspect", "topic"]],
        use_container_width=True, hide_index=True, height=260,
    )

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total requirements", len(reqs))
    kc2.metric("FR", int((reqs["fr_nfr"] == "FR").sum()))
    kc3.metric("NFR", int((reqs["fr_nfr"] == "NFR").sum()))
    kc4.metric("Topics discovered", n_topics)

    aspect_counts = reqs["aspect"].value_counts().reset_index()
    aspect_counts.columns = ["Aspect", "Requirements"]
    fig_aspect = px.bar(aspect_counts, x="Requirements", y="Aspect", orientation="h", color="Aspect",
                        color_discrete_sequence=BRAND, title="Requirements by Aspect (FR + NFR quality aspects)")
    fig_aspect.update_layout(showlegend=False, height=max(220, 60 * len(aspect_counts)),
                             yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_aspect, use_container_width=True)

    st.subheader("3. Cosine Similarity (FR/NFR Centroid Alignment)")
    st.caption("Matches the manuscript's actual classification mechanism: each requirement's LDA topic-proportion "
               "vector is compared via cosine similarity to two centroids \u2014 the mean topic vector of the FR class "
               "and of the NFR class \u2014 and the closer centroid determines its label (Eq. 1\u20132). This replaces a "
               "prior version of this chart that compared requirements to each other using TF-IDF text similarity, "
               "which was not part of the described framework.")

    centroid_fr = topic_dist[(reqs["fr_nfr"] == "FR").values].mean(axis=0)
    centroid_nfr = topic_dist[(reqs["fr_nfr"] == "NFR").values].mean(axis=0)
    n_fr = int((reqs["fr_nfr"] == "FR").sum())
    n_nfr = int((reqs["fr_nfr"] == "NFR").sum())

    if n_fr == 0 or n_nfr == 0:
        st.warning(f"Only one class is present in this dataset's current classification (FR: {n_fr}, NFR: {n_nfr}), "
                   "so an NFR (or FR) centroid can't be computed \u2014 there's nothing to average. This happens when "
                   "the classifier (ground truth for DS1, or the keyword heuristic for DS3/DS4) doesn't detect any "
                   "member of one class in this particular dataset. The FR/NFR centroid comparison isn't meaningful "
                   "here until both classes have at least one member.")
    else:
        def _cos(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        sim_fr_vals = np.array([_cos(topic_dist[i], centroid_fr) for i in range(len(reqs))])
        sim_nfr_vals = np.array([_cos(topic_dist[i], centroid_nfr) for i in range(len(reqs))])
        reqs["sim_FR"] = sim_fr_vals
        reqs["sim_NFR"] = sim_nfr_vals

        plot_src = reqs.head(30) if LARGE else reqs
        plot_idx = list(range(len(plot_src)))
        plot_df = pd.DataFrame({
            "req_id": plot_src["req_id"].values,
            "sim_FR": sim_fr_vals[plot_idx],
            "sim_NFR": sim_nfr_vals[plot_idx],
            "fr_nfr": plot_src["fr_nfr"].values,
        })

        heat_matrix = plot_df[["sim_FR", "sim_NFR"]].T
        heat_matrix.index = ["sim to FR centroid", "sim to NFR centroid"]
        heat_matrix.columns = plot_df["req_id"].values

        fig_sim = px.imshow(
            heat_matrix, color_continuous_scale="Oranges", aspect="auto",
            labels=dict(color="cosine similarity"),
            title=f"Similarity to FR vs. NFR Centroid{' (first 30 shown)' if LARGE else ''}",
            zmin=0, zmax=1,
        )
        fig_sim.update_xaxes(side="bottom", tickangle=-60)
        fig_sim.update_layout(height=420)
        st.plotly_chart(fig_sim, use_container_width=True)
        st.caption("Each column is a requirement; the two rows show its cosine similarity to the FR centroid and "
                   "to the NFR centroid. A requirement is classified by whichever row is darker (higher similarity) "
                   "for that column \u2014 columns where both rows look similarly shaded are ambiguous cases, close to "
                   "both classes.")

    st.subheader("4. ourRank Scoring (Topic-Weighted Ranking)")
    if UPLOAD_MODE:
        st.caption("Stakeholder priority defaults to the criticality of each requirement's aspect "
                   "(Security/Reliability = 5, Performance/Compliance = 4, others = 3, +1 for 'must/shall'). "
                   "You can adjust any value in the table below; your file's own values are used when provided.")
        with st.expander("\u270F\uFE0F Review / adjust priority inputs", expanded=False):
            edited = st.data_editor(
                reqs[["req_id", "requirement_text", "aspect", "stakeholder_priority", "change_frequency",
                      "historical_defects"]],
                key=f"prio_editor_{cfg['key']}", hide_index=True, use_container_width=True,
                disabled=["req_id", "requirement_text", "aspect"],
                column_config={
                    "stakeholder_priority": st.column_config.NumberColumn("Stakeholder priority (1-5)", min_value=1, max_value=5, step=1),
                    "change_frequency": st.column_config.NumberColumn("Change frequency", min_value=0, step=1),
                    "historical_defects": st.column_config.NumberColumn("Historical defects", min_value=0, step=1),
                },
            )
            for col in ["stakeholder_priority", "change_frequency", "historical_defects"]:
                reqs[col] = pd.to_numeric(edited[col], errors="coerce").fillna(0).values
    elif "fr_nfr_given" in reqs.columns:
        st.caption("Stakeholder priority / change frequency / defect history are illustrative synthetic inputs "
                   "for this public benchmark, which does not ship with project metadata.")
    st.write("Adjust weights to see how prioritization shifts in real time:")
    c1, c2, c3 = st.columns(3)
    w_stakeholder = c1.slider("Stakeholder priority weight", 0.0, 1.0, 0.4)
    w_change = c2.slider("Change frequency weight", 0.0, 1.0, 0.3)
    w_defects = c3.slider("Historical defects weight", 0.0, 1.0, 0.3)

    reqs["score"] = (
        w_stakeholder * normalize(reqs["stakeholder_priority"])
        + w_change * normalize(reqs["change_frequency"])
        + w_defects * normalize(reqs["historical_defects"])
    )
    reqs["ourRank"] = reqs["score"].rank(ascending=False).astype(int)
    ranked = reqs.sort_values("score", ascending=False)

    st.dataframe(
        ranked[["ourRank", "req_id", "requirement_text", "fr_nfr", "aspect", "topic", "score"]]
        .rename(columns={"score": "ourRank score"})
        .style.format({"ourRank score": "{:.3f}"}),
        use_container_width=True, hide_index=True, height=300,
    )

    csv_bytes = ranked[["ourRank", "req_id", "requirement_text", "fr_nfr", "aspect", "topic", "score"]].to_csv(index=False).encode()
    st.download_button("Download prioritized requirements (CSV)", csv_bytes,
                        file_name=f"{cfg['key']}_prioritized_requirements.csv", mime="text/csv")

    plot_data = ranked.head(30) if LARGE else ranked
    cbar, cdonut = st.columns([2, 1])
    with cbar:
        fig_bar = px.bar(
            plot_data, x="score", y="req_id", color="fr_nfr", orientation="h",
            color_discrete_sequence=[BRAND[0], BRAND[1]],
            title=f"ourRank Score by Requirement{' (top 30 of ' + str(len(ranked)) + ')' if LARGE else ' (sorted)'}",
            labels={"score": "ourRank score", "req_id": "Requirement", "fr_nfr": "Type"},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=460)
        st.plotly_chart(fig_bar, use_container_width=True)
    with cdonut:
        topic_counts = reqs["topic"].value_counts().sort_index()
        fig_donut = px.pie(
            names=[f"Topic {t}" for t in topic_counts.index], values=topic_counts.values,
            hole=0.55, color_discrete_sequence=BRAND,
            title="Requirements per Topic",
        )
        fig_donut.update_layout(height=460)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.session_state["ranked_reqs"] = ranked
    st.session_state["active_dataset"] = cfg["key"]

# ==========================================================================
# PHASE 2
# ==========================================================================
elif page == "Phase 2 \u2014 Test Case Selection & Prioritization":
    breadcrumb(f"Phase 2 \u00b7 {cfg['source']}")

    if not has_tc and UPLOAD_MODE:
        st.warning("Phase 2 needs test cases. Upload a **test cases file** in the panel at the top of the page "
                   "(download the sample file to see the format).")
    elif not has_tc:
        st.warning(
            f"**{dataset_name}** is a public classification benchmark and does not ship with paired "
            "test cases, so Phase 2 cannot run on it. Switch to **DS3 (EPCS Integration)** or "
            "**DS4 (Manifests Web)** in the sidebar to see the full requirement-to-test-case pipeline."
        )
    elif "ranked_reqs" not in st.session_state or st.session_state.get("active_dataset") != cfg["key"]:
        st.warning("Open **Phase 1** first (for this dataset) to generate the prioritized requirement list.")
    else:
        ranked = st.session_state["ranked_reqs"]

        testable_ids = set(tcs_raw["linked_req_id"].unique())
        ranked_testable = ranked[ranked["req_id"].isin(testable_ids)].copy()

        if len(ranked_testable) < len(ranked):
            st.caption(f"Of {len(ranked)} requirements, {len(ranked_testable)} have associated test cases "
                       f"in this dataset. Phase 2 selects among those {len(ranked_testable)}, ranked by their "
                       f"live ourRank score.")

        st.subheader("1. Domain Expert Input \u2014 Select Top-Priority Requirements")
        max_n = len(ranked_testable)
        if max_n == 0:
            st.error("None of the test cases could be aligned to a requirement. Check that the test case "
                     "descriptions relate to the requirement texts, or add a `linked_req_id` column.")
            st.stop()
        default_n = max(min(3, max_n), round(max_n * 0.6))
        if max_n <= 3:
            top_n = max_n
            st.caption(f"All {max_n} testable requirement(s) are carried into TCP.")
        else:
            top_n = st.slider("Number of top requirements to carry into TCP", min(3, max_n), max_n, default_n)
        top_reqs = ranked_testable.head(top_n)["req_id"].tolist()
        st.write("Selected requirements:", ", ".join(top_reqs))

        st.subheader("2. Requirement-Test Mapping + Greedy Heuristic TC Selection")
        priority_selected = tcs_raw[tcs_raw["linked_req_id"].isin(top_reqs)].copy()

        # --- Real redundancy detection: pairwise cosine similarity on descriptions,
        # greedily keep the higher-priority test case and drop near-duplicate lower-priority ones ---
        REDUNDANCY_THRESHOLD = 0.30
        req_rank_lookup_early = dict(zip(ranked["req_id"], ranked["ourRank"]))
        priority_selected["req_rank"] = priority_selected["linked_req_id"].map(req_rank_lookup_early)
        priority_selected = priority_selected.sort_values("req_rank").reset_index(drop=True)

        redundant_rows = []
        kept_idx = []
        if len(priority_selected) >= 2:
            vec_redund = TfidfVectorizer(stop_words="english")
            redund_src = (priority_selected["redundancy_text"] if "redundancy_text" in priority_selected.columns
                          else priority_selected["description"])
            X_redund = vec_redund.fit_transform(redund_src)
            sim_redund = cosine_similarity(X_redund)
            for i in range(len(priority_selected)):
                best_j, best_score = None, 0.0
                for j in kept_idx:
                    if sim_redund[i, j] > best_score:
                        best_j, best_score = j, sim_redund[i, j]
                if best_j is not None and best_score > REDUNDANCY_THRESHOLD:
                    redundant_rows.append({
                        "tc_id": priority_selected.iloc[i]["tc_id"],
                        "description": priority_selected.iloc[i]["description"],
                        "linked_req_id": priority_selected.iloc[i]["linked_req_id"],
                        "Reason excluded": (f"Redundant with {priority_selected.iloc[best_j]['tc_id']} "
                                             f"(cosine similarity {best_score:.2f} > {REDUNDANCY_THRESHOLD}) \u2014 "
                                             f"{priority_selected.iloc[best_j]['tc_id']} kept as the higher-priority test case"),
                    })
                else:
                    kept_idx.append(i)
        else:
            kept_idx = list(range(len(priority_selected)))

        selected = (priority_selected.iloc[kept_idx]
                    .drop(columns=["req_rank", "redundancy_text"], errors="ignore").reset_index(drop=True))
        st.dataframe(selected, use_container_width=True, hide_index=True)
        st.caption(f"{len(selected)} test cases selected out of {len(tcs_raw)} total in the pool "
                   f"(redundant/irrelevant cases excluded by construction).")

        fig_funnel = go.Figure(go.Funnel(
            y=["Total Test Case Pool", "Requirement-Relevant Candidates", "Final Prioritized Sequence (redundancy removed)"],
            x=[len(tcs_raw), len(priority_selected), len(selected)],
            marker={"color": [BRAND[2], BRAND[1], BRAND[0]]},
        ))
        fig_funnel.update_layout(title="Test Case Pool Reduction", height=320)
        st.plotly_chart(fig_funnel, use_container_width=True)

        priority_excluded = tcs_raw[~tcs_raw["tc_id"].isin(priority_selected["tc_id"])].copy()
        req_rank_lookup = dict(zip(ranked["req_id"], ranked["ourRank"]))
        total_ranked = len(ranked)
        if len(priority_excluded) > 0:
            priority_excluded["Requirement rank"] = priority_excluded["linked_req_id"].map(req_rank_lookup)
            priority_excluded["Reason excluded"] = priority_excluded.apply(
                lambda r: (f"Linked requirement {r['linked_req_id']} ranked #{int(r['Requirement rank'])} "
                           f"of {total_ranked} \u2014 below the top-{top_n} cutoff")
                if pd.notna(r["Requirement rank"])
                else ("No requirement matched this test case (text similarity too low) \u2014 not aligned"
                      if r["linked_req_id"] == "UNALIGNED"
                      else f"Linked requirement {r['linked_req_id']} has no test cases in this dataset's testable set"),
                axis=1,
            )

        redundant_df = pd.DataFrame(redundant_rows)
        total_excluded = len(priority_excluded) + len(redundant_df)

        if total_excluded > 0:
            with st.expander(f"\U0001F6AB {total_excluded} test case(s) excluded from the pool \u2014 click to see which, and why", expanded=True):
                if len(priority_excluded) > 0:
                    st.write(f"**Excluded by priority cutoff** ({len(priority_excluded)})")
                    st.dataframe(
                        priority_excluded[["tc_id", "description", "linked_req_id", "Reason excluded"]]
                        .rename(columns={"tc_id": "Test Case", "description": "Description", "linked_req_id": "Linked requirement"}),
                        use_container_width=True, hide_index=True,
                    )
                if len(redundant_df) > 0:
                    st.write(f"**Excluded as redundant** ({len(redundant_df)}) \u2014 cosine similarity to an already-kept test case exceeded {REDUNDANCY_THRESHOLD}")
                    st.dataframe(
                        redundant_df[["tc_id", "description", "linked_req_id", "Reason excluded"]]
                        .rename(columns={"tc_id": "Test Case", "description": "Description", "linked_req_id": "Linked requirement"}),
                        use_container_width=True, hide_index=True,
                    )
                st.caption("Two independent exclusion mechanisms run here: a priority cutoff (requirement rank vs. "
                           "the slider above) and a genuine redundancy check (pairwise cosine similarity between "
                           "test case descriptions, computed live \u2014 the same technique used elsewhere in this "
                           "pipeline). Whichever applies to a given test case is stated explicitly above.")
        else:
            st.success("No test cases excluded \u2014 every requirement in the pool is covered at the current cutoff, "
                       "and no redundant test cases were found.")

        st.subheader("3. Requirement\u2013Test Traceability Matrix")
        st.caption("Explicit traceability between each selected requirement and its mapped test case(s), "
                   "confirming full coverage of the prioritized requirement set.")
        trace_reqs = top_reqs
        trace_tcs = selected["tc_id"].tolist()
        trace_matrix = pd.DataFrame(0, index=trace_reqs, columns=trace_tcs)
        for _, row in selected.iterrows():
            if row["linked_req_id"] in trace_matrix.index:
                trace_matrix.loc[row["linked_req_id"], row["tc_id"]] = 1
        trace_display = trace_matrix.replace({1: "\u2713", 0: ""})
        st.dataframe(
            trace_display.style.apply(
                lambda col: ["background-color:#D9EAD3; font-weight:900; text-align:center; font-size:1.1rem"
                             if v == "\u2713" else "text-align:center" for v in col], axis=0
            ),
            use_container_width=True,
        )
        coverage_pct = (trace_matrix.sum(axis=1) > 0).mean() * 100
        st.metric("Requirement coverage", f"{coverage_pct:.0f}%",
                  help="Percentage of selected requirements with at least one traced test case.")

        if len(selected) >= 2:
            st.subheader("4. Cluster (K-Medoids)")
            if len(selected) <= 2:
                k = 2
            else:
                k = st.slider("Number of clusters (K)", 2, min(6, len(selected)), min(3, len(selected)))
            vec2 = TfidfVectorizer(stop_words="english", max_features=100)
            X2 = vec2.fit_transform(selected["description"]).toarray()
            selected["cluster"] = k_medoids(X2, k)

            if len(selected) >= 3:
                plot_df = selected.copy()
                plot_df["cluster_label"] = "Cluster " + plot_df["cluster"].astype(str)
                plot_df["size"] = 1
                fig_tree = px.treemap(
                    plot_df, path=["cluster_label", "tc_id"], values="size",
                    color="cluster_label", color_discrete_sequence=BRAND,
                    title="Test Case Clusters (grouped by similarity)",
                    hover_data={"description": True, "size": False},
                )
                fig_tree.update_traces(
                    textinfo="label",
                    textfont_size=16,
                    hovertemplate="<b>%{label}</b><br>%{customdata[0]}<extra></extra>",
                )
                fig_tree.update_layout(height=440, margin=dict(t=50, l=10, r=10, b=10))
                st.plotly_chart(fig_tree, use_container_width=True)

            st.subheader("5. Rank TC \u2014 Final Prioritized Sequence")
            req_rank_map = dict(zip(ranked["req_id"], ranked["ourRank"]))
            selected["req_rank"] = selected["linked_req_id"].map(req_rank_map)
            final_order = selected.sort_values(["req_rank", "cluster"]).reset_index(drop=True)
            final_order["TC_sequence"] = [f"TC{i+1}" for i in range(len(final_order))]
            req_info = ranked.set_index("req_id")
            final_order["fr_nfr"] = final_order["linked_req_id"].map(req_info["fr_nfr"])
            final_order["aspect"] = final_order["linked_req_id"].map(req_info["aspect"])
            rank_cut = max(1, int(np.ceil(len(ranked) / 3)))
            final_order["criticality"] = np.where(
                final_order["aspect"].isin(CRITICAL_ASPECTS) | (final_order["req_rank"] <= rank_cut), "High",
                np.where(final_order["req_rank"] <= 2 * rank_cut, "Medium", "Low"))

            st.dataframe(
                final_order[["TC_sequence", "tc_id", "description", "linked_req_id", "fr_nfr", "aspect",
                             "criticality", "cluster", "req_rank"]]
                .style.apply(lambda col: ["background-color:#F9D9C8; font-weight:800" if v == "High" else ""
                                          for v in col], subset=["criticality"]),
                use_container_width=True, hide_index=True,
            )
            n_high = int((final_order["criticality"] == "High").sum())
            first_half = final_order.head(int(np.ceil(len(final_order) / 2)))
            n_high_early = int((first_half["criticality"] == "High").sum())
            st.metric("High-criticality test cases in the first half of the sequence",
                      f"{n_high_early} of {n_high}",
                      help="High = linked to a critical aspect (Security, Reliability, Performance, Compliance) "
                           "or to a requirement in the top third of the ourRank list.")

            st.subheader("6. Aspect-Based Requirement \u2194 Prioritized Test Case Alignment")
            st.caption("Each prioritized test case is shown next to the requirement it verifies, with that "
                       "requirement's FR/NFR class, quality aspect and rank \u2014 so critical aspects are traced "
                       "to the test cases that run first.")
            align = final_order[["TC_sequence", "tc_id", "linked_req_id"]].copy()
            align["requirement_text"] = align["linked_req_id"].map(req_info["requirement_text"])
            align["fr_nfr"] = final_order["fr_nfr"].values
            align["aspect"] = final_order["aspect"].values
            align["req_rank"] = final_order["req_rank"].values
            align["criticality"] = final_order["criticality"].values
            if "alignment_score" in final_order.columns:
                align["alignment"] = (final_order["alignment_method"] + " (" +
                                      final_order["alignment_score"].map(lambda v: f"{v:.2f}") + ")").values
            st.dataframe(align, use_container_width=True, hide_index=True)
            st.download_button("Download prioritized test cases with alignment (CSV)",
                               align.merge(final_order[["tc_id", "description"]], on="tc_id").to_csv(index=False).encode(),
                               file_name=f"{cfg['key']}_prioritized_test_cases.csv", mime="text/csv")

            aspect_pos = final_order.groupby("aspect")["TC_sequence"].apply(
                lambda x: ", ".join(x)).reset_index()
            aspect_pos.columns = ["Aspect", "Executed at positions"]
            with st.expander("Where each aspect appears in the execution order"):
                st.dataframe(aspect_pos, use_container_width=True, hide_index=True)
            st.session_state["final_order"] = final_order
            st.session_state["final_order_dataset"] = cfg["key"]
        else:
            st.warning("Select more requirements above \u2014 need at least 2 matching test cases to cluster.")

# ==========================================================================
# EVALUATION
# ==========================================================================
elif page == "Evaluation \u2014 APFD":
    breadcrumb(f"Evaluation \u00b7 {cfg['source']}")

    if not has_tc and UPLOAD_MODE:
        st.warning("APFD evaluation needs test cases. Upload a **test cases file** in the panel at the top of the page.")
    elif not has_tc:
        st.warning(
            f"**{dataset_name}** has no paired test cases, so APFD evaluation isn't applicable. "
            "Switch to **DS3** or **DS4** in the sidebar."
        )
    elif "final_order" not in st.session_state or st.session_state.get("final_order_dataset") != cfg["key"]:
        st.warning("Run **Phase 1** and **Phase 2** first (for this dataset) to generate a prioritized test case sequence.")
    else:
        final_order = st.session_state["final_order"]

        if cfg["key"] == "ds4":
            st.markdown(
                '<span class="badge badge-sig">SIGNIFICANT</span> &nbsp; '
                "**DS4 real result (from the manuscript):** KS-TCP improved APFD by **+0.059** over "
                "Random Prioritization, the only statistically significant improvement across all "
                "datasets (p = 0.029, n = 14).",
                unsafe_allow_html=True,
            )
        elif cfg["key"] == "ds1":
            st.info("**DS1 published result (from the manuscript):** cluster-wise greedy ordering with "
                    "alternating-poll merge produced the sequence **038 \u2192 059 \u2192 091 \u2192 260 \u2192 010 \u2192 077 "
                    "\u2192 014 \u2192 281 \u2192 009 \u2192 036** (K = 5 clusters). Your live run below may differ slightly "
                    "depending on the ourRank weights and cluster count selected in Phase 1/2.")

        st.subheader("Executing Test Cases \u2014 Fault Detection Simulation")
        st.write("Mark which test cases (in this run) actually detected a fault:")

        all_tc_ids = final_order["tc_id"].tolist()
        fault_ids = st.multiselect("Test cases with detected faults", all_tc_ids,
                                    default=all_tc_ids[:min(3, len(all_tc_ids))])

        order = final_order["tc_id"].tolist()
        score = apfd(order, fault_ids)

        random_order = final_order["tc_id"].sample(frac=1, random_state=1).tolist()
        random_score = apfd(random_order, fault_ids)

        c1, c2, c3 = st.columns(3)
        c1.metric("APFD \u2014 Proposed order", f"{score:.3f}")
        c2.metric("APFD \u2014 Random order", f"{random_score:.3f}", delta=f"{score - random_score:+.3f}")
        c3.write(
            "APFD (Average Percentage of Faults Detected) measures how early, on average, "
            "faults are found in this execution order. Higher = faults found earlier."
        )

        st.subheader("Fault Detection Curve")
        n = len(order)
        found = set()
        pct_tc, pct_faults = [], []
        for i, tc in enumerate(order, start=1):
            if tc in fault_ids:
                found.add(tc)
            pct_tc.append(i / n * 100)
            pct_faults.append(len(found) / max(len(fault_ids), 1) * 100)

        found_r = set()
        pct_faults_r = []
        for tc in random_order:
            if tc in fault_ids:
                found_r.add(tc)
            pct_faults_r.append(len(found_r) / max(len(fault_ids), 1) * 100)

        fig_apfd = go.Figure()
        fig_apfd.add_trace(go.Scatter(
            x=[0] + pct_tc, y=[0] + pct_faults, mode="lines+markers",
            name="Proposed order", line={"color": BRAND[0], "width": 3},
            fill="tozeroy", fillcolor="rgba(189,88,44,0.12)",
        ))
        fig_apfd.add_trace(go.Scatter(
            x=[0] + pct_tc, y=[0] + pct_faults_r, mode="lines",
            name="Random order (reference)", line={"color": "#999999", "width": 2, "dash": "dash"},
        ))
        fig_apfd.update_layout(
            title="Cumulative Faults Detected vs. Test Cases Executed",
            xaxis_title="% of test suite executed", yaxis_title="% of faults detected",
            height=440, yaxis_range=[0, 105], xaxis_range=[0, 100],
        )
        st.plotly_chart(fig_apfd, use_container_width=True)

        st.subheader("Component Repository / Defect Report")
        report = final_order[["TC_sequence", "tc_id", "description", "linked_req_id", "fr_nfr", "aspect", "criticality"]].copy()
        report["fault_detected"] = report["tc_id"].isin(fault_ids)
        st.dataframe(report, use_container_width=True, hide_index=True)
