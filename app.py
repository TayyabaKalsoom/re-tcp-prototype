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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
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


# --------------------------------------------------------------------------
# FR/NFR classification: LDA topic vectors + nearest-centroid cosine rule
# (manuscript Section 3.1.1, Eq. 1-2). Semi-supervised: the LDA model needs
# no labels, but the FR and NFR centroids are the mean topic vectors of the
# labelled DS1 (PROMISE) calibration requirements. Every other dataset,
# including an uploaded one, is mapped into that fixed topic space (fold-in).
# --------------------------------------------------------------------------
CALIBRATION_KEY = "ds1"


def _fit_lda(texts, n_topics):
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30,
                                    learning_method="batch")
    lda.fit(X)
    return vec, lda, lda.transform(X)


def _class_centroids(theta, labels):
    labels = np.asarray(labels)
    return theta[labels == "FR"].mean(axis=0), theta[labels == "NFR"].mean(axis=0)


def _cos_to(theta, c):
    num = theta @ c
    den = np.linalg.norm(theta, axis=1) * np.linalg.norm(c)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def centroid_rule(theta, c_fr, c_nfr):
    """Eq. (2): label = argmax over {FR, NFR} of CosSim(theta, centroid)."""
    s_fr, s_nfr = _cos_to(theta, c_fr), _cos_to(theta, c_nfr)
    return np.where(s_nfr > s_fr, "NFR", "FR"), s_fr, s_nfr


def _calibration_df():
    cal = next(d for d in DATASETS.values() if d["key"] == CALIBRATION_KEY)
    df = load_csv(cal["req_csv"])
    return df[df["fr_nfr_given"].isin(["FR", "NFR"])].reset_index(drop=True)


@st.cache_resource(show_spinner="Fitting the DS1 calibration model (LDA + FR/NFR centroids)...")
def calibration_model(n_topics):
    df = _calibration_df()
    vec, lda, theta = _fit_lda(df["requirement_text"].astype(str).tolist(), n_topics)
    c_fr, c_nfr = _class_centroids(theta, df["fr_nfr_given"].values)
    return {"vec": vec, "lda": lda, "c_fr": c_fr, "c_nfr": c_nfr,
            "n_fr": int((df["fr_nfr_given"] == "FR").sum()), "n_nfr": int((df["fr_nfr_given"] == "NFR").sum())}


@st.cache_data(show_spinner="Classifying DS1 on held-out folds...")
def ds1_held_out_classification(n_topics, n_splits=5):
    """DS1 is the calibration set, so each requirement is classified by a model (LDA + centroids)
    fitted on the other folds only, never on itself."""
    df = _calibration_df()
    texts = df["requirement_text"].astype(str).tolist()
    labels = df["fr_nfr_given"].values
    pred = np.empty(len(df), dtype=object)
    s_fr = np.zeros(len(df))
    s_nfr = np.zeros(len(df))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(texts, labels):
        vec, lda, th_tr = _fit_lda([texts[i] for i in tr], n_topics)
        c_fr, c_nfr = _class_centroids(th_tr, labels[tr])
        lab, a, b = centroid_rule(lda.transform(vec.transform([texts[i] for i in te])), c_fr, c_nfr)
        pred[te], s_fr[te], s_nfr[te] = lab, a, b
    return pd.DataFrame({"req_id": df["req_id"].values, "fr_nfr": pred, "sim_FR": s_fr, "sim_NFR": s_nfr,
                         "ground_truth": labels})


def vocabulary_coverage(vec, texts):
    """Share of (unigram, stop-word-free) tokens that exist in the calibration vocabulary."""
    analyzer = CountVectorizer(stop_words="english").build_analyzer()
    vocab = {t for t in vec.vocabulary_ if " " not in t}
    toks = [t for txt in texts for t in analyzer(str(txt))]
    return (sum(t in vocab for t in toks) / len(toks)) if toks else 0.0


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


def fault_detecting_tcs(df: pd.DataFrame):
    """Return (sorted list of fault-detecting tc_ids, description of the format found).

    Accepts any of:
      * execution history  - columns tc_id + verdict (FAIL / FAILED / ERROR = detected),
                             or tc_id + methods_failed (> 0 = detected)
      * fault matrix       - tc_id + one 0/1 column per fault (any 1 = detected)
      * faults list        - a detecting_tests column with ';'-separated tc_ids
      * test case list     - tc_id + detected_fault (1/0)
    """
    cols = {c.strip().lower(): c for c in df.columns}
    tc_col = cols.get("tc_id") or cols.get("test_id") or cols.get("test_case")

    if "detecting_tests" in cols:
        ids = set()
        for cell in df[cols["detecting_tests"]].dropna().astype(str):
            ids.update(t.strip() for t in cell.replace(",", ";").split(";") if t.strip())
        return sorted(ids), "faults list (detecting_tests column)"

    if tc_col is None:
        return None, "no tc_id column found"

    tc = df[tc_col].astype(str).str.strip()
    if "verdict" in cols:
        v = df[cols["verdict"]].astype(str).str.strip().str.upper()
        mask = v.isin(["FAIL", "FAILED", "FAILURE", "ERROR", "KILLED", "DETECTED"])
        return sorted(tc[mask].unique()), "execution history (verdict = FAIL)"
    if "methods_failed" in cols:
        mask = pd.to_numeric(df[cols["methods_failed"]], errors="coerce").fillna(0) > 0
        return sorted(tc[mask].unique()), "execution history (methods_failed > 0)"
    if "detected_fault" in cols:
        mask = pd.to_numeric(df[cols["detected_fault"]], errors="coerce").fillna(0) > 0
        return sorted(tc[mask].unique()), "detected_fault column"

    fault_cols = [c for c in df.columns if c != tc_col]
    numeric = df[fault_cols].apply(pd.to_numeric, errors="coerce")
    fault_cols = [c for c in fault_cols if numeric[c].notna().all() and set(numeric[c].unique()) <= {0, 1}]
    if fault_cols:
        mask = numeric[fault_cols].sum(axis=1) > 0
        return sorted(tc[mask].unique()), f"fault matrix ({len(fault_cols)} fault columns)"
    return None, "no verdict, detecting_tests or 0/1 fault columns found"


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
TC_LINK_CANDIDATES = ["linked_req_id", "linked_req_ids", "req_id", "requirement_id", "linked_requirement",
                      "requirements_covered", "requirement"]
TC_OBLIGATION_CANDIDATES = ["obligation", "obligations", "verification_obligation"]
TC_FAULT_CANDIDATES = ["detected_fault", "fault_detected", "fault", "faults", "failed"]
TC_PAST_FAIL_CANDIDATES = ["past_failures", "historical_failures", "failure_count", "past_fails"]
TC_COVERAGE_CANDIDATES = ["lines_covered", "code_coverage", "coverage", "statements_covered"]


OBLIGATION_TITLE_THRESHOLD = 0.5
_FAILURE_WORDS = (r"\b(fail|fails|failed|failure|invalid|incorrect|wrong|unauthori[sz]ed|declined|reject|rejected|"
                  r"denied|deny|error|blocked|locked|lockout|cannot|can't|not|without permission)\b|\bnon-(registered|authori[sz]ed|epcs)\b")
_BOUNDARY_WORDS = (r"\d|\b(within|under|over|limit|limits|maximum|minimum|max|min|threshold|timeout|concurrent|"
                   r"seconds?|minutes?|boundary|exceed|exceeds)\b")


def obligation_kind(text):
    """Kind of verification obligation a test case exercises: failure path, boundary, or nominal behaviour."""
    import re
    t = test_title(text).lower()
    if re.search(_FAILURE_WORDS, t):
        return "failure"
    if re.search(_BOUNDARY_WORDS, t):
        return "boundary"
    return "nominal"


def test_title(text):
    """The test's own name: text before 'Preconditions' or a '(section)' reference, without 'Test case:'."""
    import re
    s = re.split(r"\bPre-?conditions?\b|\(\d", str(text), maxsplit=1, flags=re.I)[0]
    return re.sub(r"^\s*test\s*case\s*:\s*", "", s, flags=re.I).strip() or str(text)


def split_req_ids(value):
    """A test case may verify several requirements: 'R1;R2', 'R1, R2' or 'R1|R2'."""
    import re
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [x.strip() for x in re.split(r"[;,|]", str(value)) if x.strip() and x.strip().lower() != "nan"]

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
    obl_col = _pick_col(df, TC_OBLIGATION_CANDIDATES)
    fault_col = _pick_col(df, TC_FAULT_CANDIDATES)
    link_vals = (df[link_col].fillna("").astype(str).str.strip().reset_index(drop=True)
                 if link_col is not None else pd.Series([""] * len(out)))

    vec = TfidfVectorizer(stop_words="english")
    vec.fit(pd.concat([reqs["requirement_text"], out["description"]]))
    sim = cosine_similarity(vec.transform(out["description"]), vec.transform(reqs["requirement_text"]))
    req_ids = reqs["req_id"].tolist()
    linked, scores, methods = [], [], []
    file_has_links = bool((link_vals.map(lambda v: bool(split_req_ids(v))) if len(link_vals) else pd.Series([], dtype=bool)).any())
    for i in range(len(out)):
        given = str(link_vals.iloc[i]) if i < len(link_vals) and pd.notna(link_vals.iloc[i]) else ""
        given_ids = [g for g in split_req_ids(given) if g in req_ids]
        if given_ids:
            j = req_ids.index(given_ids[0])
            linked.append(";".join(given_ids)); scores.append(float(sim[i, j])); methods.append("Given in file")
        elif file_has_links:
            # The file gives traceability links: a blank link means the test verifies no listed requirement
            # (e.g. framework or infrastructure tests), so it is not auto-aligned and is dropped as irrelevant.
            j = int(sim[i].argmax()) if len(req_ids) else 0
            linked.append("UNALIGNED"); scores.append(float(sim[i, j]) if len(req_ids) else 0.0)
            methods.append("Not linked in file")
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
    out["obligation"] = (df[obl_col].fillna("").astype(str).str.strip().reset_index(drop=True)
                         if obl_col is not None else "")
    out["detected_fault"] = (df[fault_col].astype(str).str.strip().str.lower()
                             .isin(["1", "true", "yes", "y", "x", "fault", "failed"]).reset_index(drop=True)
                             if fault_col is not None else False)
    cols = ["tc_id", "description", "linked_req_id", "alignment_score", "alignment_method",
            "obligation", "detected_fault"]
    # Optional test-level regression history: past failures of each test and the amount of code it executes,
    # both measured on earlier runs (available before the new version is tested).
    for name, cands in [("past_failures", TC_PAST_FAIL_CANDIDATES), ("lines_covered", TC_COVERAGE_CANDIDATES)]:
        col = _pick_col(df, cands)
        if col is not None:
            out[name] = pd.to_numeric(df[col], errors="coerce").fillna(0).reset_index(drop=True)
            cols.append(name)
    return out[cols]


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

    # Fixed calibration model (fitted on labelled DS1); this dataset is folded into its topic space.
    model = calibration_model(n_topics)
    texts = reqs_raw["requirement_text"].astype(str).tolist()
    topic_dist = model["lda"].transform(model["vec"].transform(texts))
    reqs = reqs_raw.copy()
    reqs["topic"] = topic_dist.argmax(axis=1)

    is_calibration_ds = (not UPLOAD_MODE) and cfg["key"] == CALIBRATION_KEY
    if is_calibration_ds:
        held_out = ds1_held_out_classification(n_topics).drop_duplicates("req_id").set_index("req_id")
        for col in ["fr_nfr", "sim_FR", "sim_NFR"]:
            reqs[col] = reqs["req_id"].map(held_out[col])
        truth = reqs["fr_nfr_given"]
        acc = float((reqs["fr_nfr"] == truth).mean())
        baseline = float(truth.value_counts(normalize=True).max())
        st.caption("FR/NFR labels come from the LDA nearest-centroid rule (Eq. 1–2). DS1 is the labelled "
                   "calibration set, so each requirement is classified by a model whose LDA and centroids were "
                   "fitted on the other four folds only (stratified 5-fold).")
        a1, a2 = st.columns(2)
        a1.metric("Agreement with ground-truth labels", f"{acc:.1%}")
        a2.metric("Majority-class baseline", f"{baseline:.1%}")
    else:
        labels, s_fr, s_nfr = centroid_rule(topic_dist, model["c_fr"], model["c_nfr"])
        reqs["fr_nfr"], reqs["sim_FR"], reqs["sim_NFR"] = labels, s_fr, s_nfr
        cov = vocabulary_coverage(model["vec"], texts)
        st.caption("FR/NFR labels come from the LDA nearest-centroid rule (Eq. 1–2): each requirement's "
                   "topic vector is compared by cosine similarity with the FR and NFR centroids learned from the "
                   f"labelled DS1 (PROMISE) calibration set ({model['n_fr']} FR, {model['n_nfr']} NFR), and the "
                   f"closer centroid gives the label. {cov:.0%} of this dataset's words are in the calibration "
                   "vocabulary; unknown words are ignored. The quality aspect (Security, Performance, ...) is "
                   "detected separately from ISO/IEC 25010 keywords.")

    if "aspect" not in reqs.columns:
        reqs["aspect"] = reqs["requirement_text"].apply(classify_aspect)
    # The quality aspect (ISO/IEC 25010 keywords) is kept as an independent signal: a requirement the
    # centroid rule labels FR can still carry a Security/Performance/... aspect. Only an NFR with no
    # keyword aspect is relabelled, so it does not read as "Functional".
    reqs.loc[(reqs["fr_nfr"] == "NFR") & (reqs["aspect"] == "Functional"), "aspect"] = "NFR (general)"

    class_cols = ["req_id", "requirement_text", "fr_nfr", "aspect", "topic", "sim_FR", "sim_NFR"]
    if is_calibration_ds:
        class_cols.insert(3, "fr_nfr_given")
    st.dataframe(
        reqs[class_cols].rename(columns={"fr_nfr_given": "ground_truth"})
        .style.format({"sim_FR": "{:.3f}", "sim_NFR": "{:.3f}"}),
        use_container_width=True, hide_index=True, height=260,
    )

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total requirements", len(reqs))
    kc2.metric("FR", int((reqs["fr_nfr"] == "FR").sum()))
    kc3.metric("NFR", int((reqs["fr_nfr"] == "NFR").sum()))
    kc4.metric("Topics discovered", n_topics)

    aspect_counts = reqs.groupby(["aspect", "fr_nfr"]).size().reset_index(name="Requirements")
    aspect_counts.columns = ["Aspect", "FR/NFR (centroid rule)", "Requirements"]
    fig_aspect = px.bar(aspect_counts, x="Requirements", y="Aspect", orientation="h",
                        color="FR/NFR (centroid rule)", color_discrete_map={"FR": ORANGE, "NFR": NAVY},
                        title="Quality aspect (keywords) vs. FR/NFR label (centroid rule)")
    fig_aspect.update_layout(barmode="stack", height=max(240, 60 * aspect_counts["Aspect"].nunique()),
                             yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_aspect, use_container_width=True)
    st.caption("Two independent signals: the bar's row is the quality aspect detected from ISO/IEC 25010 keywords; "
               "its colour is the FR/NFR label from the LDA nearest-centroid rule. An orange segment on a quality "
               "aspect (e.g. Security) is a requirement the centroid rule called FR although its wording names a "
               "quality concern — the cases where the two signals disagree.")

    st.subheader("3. Cosine Similarity (FR/NFR Centroid Alignment)")
    st.caption("Matches the manuscript's actual classification mechanism: each requirement's LDA topic-proportion "
               "vector is compared via cosine similarity to two centroids \u2014 the mean topic vector of the FR class "
               "and of the NFR class, both learned from the labelled DS1 calibration set \u2014 and the closer "
               "centroid determines its label (Eq. 1\u20132). These are the same similarities that produced the "
               "FR/NFR column above.")

    sim_fr_vals = reqs["sim_FR"].to_numpy(dtype=float)
    sim_nfr_vals = reqs["sim_NFR"].to_numpy(dtype=float)

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
    reqs["ourRank"] = reqs["score"].round(6).rank(ascending=False, method="min").astype(int)
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

        tcs_raw = tcs_raw.copy()
        tcs_raw["covers"] = tcs_raw["linked_req_id"].map(split_req_ids)
        testable_ids = {r for lst in tcs_raw["covers"] for r in lst}
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
        req_rank_lookup = dict(zip(ranked["req_id"], ranked["ourRank"]))
        top_set = set(top_reqs)
        tcs_raw["in_scope"] = tcs_raw["covers"].map(lambda lst: [r for r in lst if r in top_set])
        tcs_raw["best_rank"] = tcs_raw["in_scope"].map(
            lambda lst: min((req_rank_lookup.get(r, 10**6) for r in lst), default=10**6))
        candidates = tcs_raw[tcs_raw["in_scope"].map(len) > 0].reset_index(drop=True)
        candidates["file_pos"] = range(len(candidates))

        # --- Verification obligations (manuscript Section 3.2.1) ---------------------------------
        # Each requirement r has obligations: one nominal path, one per boundary/threshold and one per
        # failure path. A test case exercises (r, obligation) for every in-scope requirement it traces to.
        # The obligation label comes from the file's `obligation` column when given (manual annotation);
        # otherwise it is derived: the kind (nominal / boundary / failure) from the test wording, and tests
        # of the same requirement and kind whose titles are near-identical (cosine > 0.5) share one obligation.
        candidates["kind"] = candidates["description"].map(obligation_kind)
        candidates["title"] = candidates["description"].map(test_title)
        manual = candidates["obligation"].astype(str).str.strip() != "" if "obligation" in candidates.columns \
            else pd.Series(False, index=candidates.index)
        title_sim = (cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(candidates["title"]))
                     if len(candidates) >= 2 else np.ones((len(candidates), len(candidates))))
        obligations = {i: set() for i in candidates.index}
        for r in top_reqs:
            idx = [i for i in candidates.sort_values(["best_rank", "file_pos"]).index if r in candidates.at[i, "in_scope"]]
            groups = []  # (kind, representative index)
            for i in idx:
                if manual.iloc[i]:
                    obligations[i].add((r, candidates.at[i, "obligation"]))
                    continue
                kind = candidates.at[i, "kind"]
                g = next((gi for gi, (k, rep_i) in enumerate(groups)
                          if k == kind and title_sim[i, rep_i] > OBLIGATION_TITLE_THRESHOLD), None)
                if g is None:
                    groups.append((kind, i)); g = len(groups) - 1
                obligations[i].add((r, f"{kind} #{sum(1 for k, _ in groups[:g + 1] if k == kind)}"))

        # --- Deterministic greedy heuristic over obligations --------------------------------------
        uncovered = set().union(*obligations.values()) if obligations else set()
        chosen, covered_by = [], {}
        while uncovered:
            gain = {i: len(obligations[i] & uncovered) for i in candidates.index if i not in chosen}
            best = max(gain.values(), default=0)
            if best == 0:
                break
            pick = min((i for i, g in gain.items() if g == best),
                       key=lambda i: (candidates.at[i, "best_rank"], candidates.at[i, "file_pos"]))
            chosen.append(pick)
            for ob in obligations[pick] & uncovered:
                covered_by[ob] = candidates.at[pick, "tc_id"]
            uncovered -= obligations[pick]

        redundant_rows = []
        for i in candidates.index:
            if i in chosen:
                continue
            by = sorted({covered_by[ob] for ob in obligations[i] if ob in covered_by})
            obl_txt = "; ".join(f"{r} {o}" for r, o in sorted(obligations[i]))
            redundant_rows.append({
                "tc_id": candidates.at[i, "tc_id"], "description": candidates.at[i, "description"],
                "linked_req_id": candidates.at[i, "linked_req_id"],
                "Reason excluded": f"Redundant: every obligation it checks ({obl_txt}) is already checked by "
                                   f"{', '.join(by)}",
            })

        def _primary(lst):
            return min(lst, key=lambda r: req_rank_lookup.get(r, 10**6))

        selected = candidates.loc[chosen].copy()
        selected["covers"] = selected["in_scope"].map(lambda lst: ";".join(lst))
        selected["linked_req_id"] = selected["in_scope"].map(_primary)
        selected["obligations"] = [ "; ".join(f"{r} {o}" for r, o in sorted(obligations[i])) for i in chosen]
        selected = selected.reset_index(drop=True)
        show_cols = ["tc_id", "description", "covers", "obligations"]
        st.dataframe(selected[show_cols].rename(columns={"covers": "requirements verified (in scope)"}),
                     use_container_width=True, hide_index=True)
        st.caption(f"{len(selected)} test cases selected out of {len(tcs_raw)} total in the pool. Greedy heuristic: "
                   "at each step the test case checking the most still-unchecked verification obligations is "
                   "selected (ties \u2192 highest-ranked requirement), so every obligation of every in-scope "
                   "requirement stays covered.")

        fig_funnel = go.Figure(go.Funnel(
            y=["Total Test Case Pool", "Requirement-Relevant Candidates", "Final Selected Suite (redundancy removed)"],
            x=[len(tcs_raw), len(candidates), len(selected)],
            marker={"color": [BRAND[2], BRAND[1], BRAND[0]]},
        ))
        fig_funnel.update_layout(title="Test Case Pool Reduction", height=320)
        st.plotly_chart(fig_funnel, use_container_width=True)

        priority_excluded = tcs_raw[tcs_raw["in_scope"].map(len) == 0].copy()
        total_ranked = len(ranked)
        if len(priority_excluded) > 0:
            def _irrelevant_reason(r):
                if r["linked_req_id"] == "UNALIGNED" or not r["covers"]:
                    return "No requirement matched this test case (text similarity too low) \u2014 not aligned"
                ranks = [req_rank_lookup[x] for x in r["covers"] if x in req_rank_lookup]
                if ranks:
                    return (f"Verifies {', '.join(r['covers'])} \u2014 best rank #{min(ranks)} of {total_ranked}, "
                            f"below the top-{top_n} cutoff")
                return f"Linked requirement(s) {', '.join(r['covers'])} not in this dataset's testable set"
            priority_excluded["Reason excluded"] = priority_excluded.apply(_irrelevant_reason, axis=1)

        redundant_df = pd.DataFrame(redundant_rows)
        total_excluded = len(priority_excluded) + len(redundant_df)

        if total_excluded > 0:
            with st.expander(f"\U0001F6AB {total_excluded} test case(s) excluded from the pool \u2014 click to see which, and why", expanded=True):
                if len(priority_excluded) > 0:
                    st.write(f"**Excluded as irrelevant** ({len(priority_excluded)}) \u2014 none of the requirements they verify is in the top-{top_n}")
                    st.dataframe(
                        priority_excluded[["tc_id", "description", "linked_req_id", "Reason excluded"]]
                        .rename(columns={"tc_id": "Test Case", "description": "Description", "linked_req_id": "Linked requirement(s)"}),
                        use_container_width=True, hide_index=True,
                    )
                if len(redundant_df) > 0:
                    st.write(f"**Excluded as redundant** ({len(redundant_df)}) \u2014 every verification obligation they check is already checked by a selected test case")
                    st.dataframe(
                        redundant_df[["tc_id", "description", "linked_req_id", "Reason excluded"]]
                        .rename(columns={"tc_id": "Test Case", "description": "Description", "linked_req_id": "Linked requirement(s)"}),
                        use_container_width=True, hide_index=True,
                    )
                st.caption("Obligations: one nominal path, one per boundary/threshold and one per failure path of each "
                           "requirement. Taken from the file's `obligation` column when given; otherwise derived from "
                           "the test wording (failure words such as 'fails', 'invalid', 'non-registered'; boundary words "
                           "such as numbers, 'within', 'concurrent'), with near-identical titles (cosine > "
                           f"{OBLIGATION_TITLE_THRESHOLD}) of the same requirement and kind counted as one obligation.")
        else:
            st.success("No test cases excluded \u2014 every requirement in the pool is covered at the current cutoff, "
                       "and every test case checks at least one obligation no other test case checks.")

        st.subheader("3. Requirement\u2013Test Traceability Matrix")
        st.caption("Explicit traceability between each selected requirement and its mapped test case(s), "
                   "confirming full coverage of the prioritized requirement set.")
        trace_reqs = top_reqs
        trace_tcs = selected["tc_id"].tolist()
        trace_matrix = pd.DataFrame(0, index=trace_reqs, columns=trace_tcs)
        for _, row in selected.iterrows():
            for r in split_req_ids(row["covers"]):
                if r in trace_matrix.index:
                    trace_matrix.loc[r, row["tc_id"]] = 1
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
            st.caption("Order: High-criticality test cases first (they verify a requirement with a critical "
                       "quality aspect or one in the top third of ourRank), then Medium, then Low; within each "
                       "level by ourRank of the requirement, then by K-Medoids cluster. Coverage step: a test "
                       "case that verifies a requirement not yet verified runs before test cases that only "
                       "re-verify requirements already covered.")
            if {"past_failures", "lines_covered"} <= set(selected.columns):
                st.caption("Test-level regression history found (past_failures, lines_covered): before the coverage "
                           "step, test cases are ranked by 0.4 \u00d7 requirement-driven position + 0.3 \u00d7 past "
                           "failures + 0.3 \u00d7 code executed.")
            req_rank_map = dict(zip(ranked["req_id"], ranked["ourRank"]))
            selected["req_rank"] = selected["linked_req_id"].map(req_rank_map)
            req_info = ranked.set_index("req_id")
            # Aspect-based criticality: a test case is High when any requirement it verifies has a critical
            # quality aspect (Security, Reliability, Performance, Legal/Compliance) or is in the top third of
            # ourRank; Medium for the middle third; Low otherwise. Critical tests run first, then ourRank,
            # then K-Medoids cluster.
            rank_cut = max(1, int(np.ceil(len(ranked) / 3)))
            covers_lists = selected["covers"].map(lambda s: split_req_ids(s) or [])
            crit_aspect = covers_lists.map(lambda lst: any(req_info["aspect"].get(r) in CRITICAL_ASPECTS for r in lst))
            selected["criticality"] = np.where(
                crit_aspect | (selected["req_rank"] <= rank_cut), "High",
                np.where(selected["req_rank"] <= 2 * rank_cut, "Medium", "Low"))
            selected["_crit_order"] = selected["criticality"].map({"High": 0, "Medium": 1, "Low": 2})
            tiered = (selected.sort_values(["_crit_order", "req_rank", "cluster"], kind="stable")
                      .drop(columns="_crit_order").reset_index(drop=True))
            # Coverage step (paper Section 3.2.3): walking the tiered order, a test case that verifies at least
            # one requirement not yet verified runs first; test cases that only re-verify requirements are
            # deferred until every requirement has been verified once, then run in the same order.
            hist_cols = [c for c in ("past_failures", "lines_covered") if c in tiered.columns]
            use_history = len(hist_cols) == 2
            if use_history:
                # Test-level regression history (optional columns): score = 0.4 x requirement-driven position
                # + 0.3 x past failures + 0.3 x code executed, each scaled to 0..1; the coverage step below
                # is then applied to this order, so every requirement is still verified as early as possible.
                def _scale(v):
                    v = v.astype(float); rng_ = v.max() - v.min()
                    return (v - v.min()) / rng_ if rng_ > 0 else v * 0
                n_t = len(tiered)
                tiered["req_driven_score"] = 1 - np.arange(n_t) / max(n_t, 1)
                tiered["tps"] = (0.4 * tiered["req_driven_score"] + 0.3 * _scale(tiered["past_failures"])
                                 + 0.3 * _scale(tiered["lines_covered"]))
                tiered = tiered.sort_values("tps", ascending=False, kind="stable").reset_index(drop=True)
            verified, first_pass, deferred = set(), [], []
            for i, lst in enumerate(tiered["covers"].map(lambda s: set(split_req_ids(s) or []))):
                if lst - verified:
                    first_pass.append(i); verified |= lst
                else:
                    deferred.append(i)
            final_order = tiered.iloc[first_pass + deferred].reset_index(drop=True)
            final_order["coverage_step"] = ["New requirement"] * len(first_pass) + ["Re-verifies (deferred)"] * len(deferred)
            final_order["TC_sequence"] = [f"TC{i+1}" for i in range(len(final_order))]
            final_order["fr_nfr"] = final_order["linked_req_id"].map(req_info["fr_nfr"])
            final_order["aspect"] = final_order["linked_req_id"].map(req_info["aspect"])

            st.dataframe(
                final_order[["TC_sequence", "tc_id", "description", "linked_req_id", "fr_nfr", "aspect",
                             "criticality", "coverage_step", "cluster", "req_rank"]]
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
        file_faults = (final_order.loc[final_order["detected_fault"].astype(bool), "tc_id"].tolist()
                       if "detected_fault" in final_order.columns else [])
        fault_source = "the `detected_fault` column of your test case file" if file_faults else ""

        exec_file = st.file_uploader(
            "Upload execution history / fault data to select fault-detecting test cases automatically",
            type=["csv", "xlsx", "xls"], key="up_exec",
            help="Execution history (tc_id + verdict, FAIL = detected), a fault matrix (tc_id + one 0/1 column "
                 "per fault) or a faults list (detecting_tests column). Large files are fine.")
        if exec_file is not None:
            try:
                detected, fmt = fault_detecting_tcs(read_uploaded_table(exec_file))
            except Exception as e:  # noqa: BLE001
                detected, fmt = None, f"could not read the file ({e})"
            if detected is None:
                st.error(f"Couldn't find fault data in **{exec_file.name}**: {fmt}.")
            else:
                in_order = [t for t in all_tc_ids if t in set(detected)]
                missing = [t for t in detected if t not in set(all_tc_ids)]
                file_faults = in_order
                fault_source = f"**{exec_file.name}** — read as {fmt}"
                if missing:
                    shown = ", ".join(missing[:15]) + (" …" if len(missing) > 15 else "")
                    st.warning(f"{len(missing)} fault-detecting test case(s) in the file are not in the "
                               f"prioritized sequence (deselected or unmatched IDs) and were ignored: {shown}")

        fault_ids = st.multiselect("Test cases with detected faults", all_tc_ids, default=file_faults,
                                   key=f"fault_ids_{hash(tuple(file_faults))}")
        if file_faults:
            st.caption(f"{len(file_faults)} of {len(all_tc_ids)} test case(s) auto-selected from {fault_source}. "
                       "You can still add or remove test cases by hand.")
        else:
            st.info("No fault data yet. Upload your execution history above, mark the test cases that actually "
                    "detected a fault by hand, or add a `detected_fault` column with 1/0 to your test case file. APFD is only meaningful with real "
                    "fault data: faults must not be chosen from the proposed order itself.")

        order = final_order["tc_id"].tolist()
        score = apfd(order, fault_ids)

        random_order = final_order["tc_id"].sample(frac=1, random_state=1).tolist()
        _rng = np.random.default_rng(1)
        random_score = (float(np.mean([apfd(list(_rng.permutation(order)), fault_ids) for _ in range(100)]))
                        if fault_ids else 0.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("APFD \u2014 Proposed order", f"{score:.3f}")
        c2.metric("APFD \u2014 Random order (mean of 100)", f"{random_score:.3f}", delta=f"{score - random_score:+.3f}")
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
