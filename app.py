import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    layout="wide",
    page_icon="🔍",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.15rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(135deg, rgba(224,92,42,0.16), rgba(91,143,201,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.2rem 1.25rem;
        border-radius: 22px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(20, 22, 28, 0.92);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        min-height: 118px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aab3c5;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .metric-sub {
        font-size: 0.84rem;
        color: #8f98aa;
    }
    .section-card {
        background: rgba(16, 18, 24, 0.96);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        margin-top: 0.9rem;
    }
    .risk-title {
        font-size: 1.04rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .small-muted { color: #9aa3b2; font-size: 0.88rem; }
    .card {
        background: linear-gradient(180deg, rgba(35,38,48,0.98), rgba(22,24,31,0.98));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 0.95rem 0.95rem 0.85rem 0.95rem;
        height: 100%;
    }
    .card-topline { display:flex; justify-content:space-between; gap:0.75rem; margin-bottom:0.45rem; }
    .pill {
        font-size: 0.74rem;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        background: rgba(224,92,42,0.16);
        color: #ffb199;
        border: 1px solid rgba(224,92,42,0.25);
        white-space: nowrap;
    }
    .pill-blue {
        background: rgba(91,143,201,0.16);
        color: #c1daf7;
        border: 1px solid rgba(91,143,201,0.25);
    }
    .ticket-id { font-weight: 700; font-size: 1rem; margin-bottom: 0.15rem; }
    .ticket-meta { font-size: 0.82rem; color: #a4adbc; line-height: 1.45; }
    .ticket-desc {
        font-size: 0.82rem;
        color: #d7dce4;
        margin-top: 0.5rem;
        line-height: 1.45;
        max-height: 92px;
        overflow: hidden;
    }
    .rank-name { font-weight: 600; }
    .bar-wrap {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #e05c2a, #f5a35e);
    }
    .bar-fill-blue {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #5b8fc9, #7dc1ff);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Constants
# ---------------------------
PMAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
RPMAP = {v: k for k, v in PMAP.items()}
BASE = "microsoft/deberta-v3-small"
MDL = Path("models/sia_model")
OUT = Path("outputs")

ESCA_EV = [
    (r"\bfraud\w*\b", "fraud_indicator", 0.35),
    (r"\bphish\w*\b", "security_threat", 0.35),
    (r"\bhack\w*\b", "security_threat", 0.30),
    (r"\bstolen\b", "security_threat", 0.30),
    (r"\bunauthori[sz]ed\b", "security_threat", 0.28),
    (r"\bdata\s+breach\b", "data_risk", 0.32),
    (r"\bdata\s+loss\b", "data_risk", 0.30),
    (r"\bcrash\w*\b", "system_failure", 0.22),
    (r"\bnot\s+(loading|working|responding)\b", "functional_failure", 0.20),
    (r"\blocke?d\s+out\b", "access_blocked", 0.24),
    (r"\bpayment\s+fail\w*\b", "payment_failure", 0.22),
    (r"\bcompromised\b", "account_risk", 0.28),
    (r"\bimmediately\b", "urgency", 0.14),
    (r"\burgent\b", "urgency", 0.14),
    (r"\bransomware\b", "security_threat", 0.40),
]

DEESC_EV = [
    (r"\bhow\s+do\s+i\b", "informational", -0.14),
    (r"\bwhere\s+is\b", "informational", -0.14),
    (r"\bfeature\s+request\b", "feature_req", -0.16),
    (r"\bheadquarters\b", "general_query", -0.20),
    (r"\broadmap\b", "general_query", -0.16),
]

RT_BENCH = {
    ("Fraud", "Critical"): 4,
    ("Fraud", "High"): 12,
    ("Technical", "Critical"): 5,
    ("Technical", "High"): 18,
    ("Technical", "Medium"): 38,
    ("Technical", "Low"): 50,
    ("Billing", "Critical"): 6,
    ("Billing", "High"): 20,
    ("Billing", "Medium"): 42,
    ("Billing", "Low"): 52,
    ("Account", "High"): 22,
    ("Account", "Medium"): 40,
    ("Account", "Low"): 50,
    ("General Inquiry", "Medium"): 35,
    ("General Inquiry", "Low"): 45,
}

CAT_SEV = {
    "Fraud": {"exp": "Critical", "w": 0.28, "note": "Fraud carries inherent security risk"},
    "Technical": {"exp": "High", "w": 0.18, "note": "Technical failures impact availability"},
    "Account": {"exp": "Medium", "w": 0.12, "note": "Account issues affect user access"},
    "Billing": {"exp": "Medium", "w": 0.10, "note": "Billing issues have financial impact"},
    "General Inquiry": {"exp": "Low", "w": -0.15, "note": "General inquiries are informational"},
}

# ---------------------------
# Helper Functions
# ---------------------------
def rt_tier(h):
    if h <= 10:
        return "FAST"
    if h <= 45:
        return "MID"
    return "SLOW"


def make_input(row):
    rt = float(row.get("Resolution_Time_Hours", 30))
    return (
        f"[SUBJ] {row['Ticket_Subject']} [BODY] {row['Ticket_Description']} "
        f"| cat:{row['Issue_Category']} | ch:{row.get('Ticket_Channel', 'Unknown')} "
        f"| rt:{rt_tier(rt)} | pri:{row['Priority_Level']}"
    )


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    best = MDL / "best"
    if not best.exists():
        return None, None, 0.5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(best))
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base, str(best)).float().to(device)
    model.eval()
    tf = MDL / "threshold.npy"
    thr = float(np.load(str(tf))[0]) if tf.exists() else 0.5
    return tok, model, thr


def predict(texts, tok, model, thr):
    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(texts), 32):
        enc = tok(
            texts[i:i + 32],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        probs += torch.softmax(out.logits.float(), -1)[:, 1].cpu().tolist()
    probs = np.array(probs)
    return probs, (probs >= thr).astype(int)


def dir_score(row):
    t = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    s = sum(w for _, _, w in ESCA_EV + DEESC_EV if re.search(_, t))
    cs = CAT_SEV.get(row["Issue_Category"], {"exp": "Medium", "w": 0.05})
    if PMAP.get(cs["exp"], 1) > PMAP.get(row["Priority_Level"], 1):
        s += abs(cs["w"])
    elif PMAP.get(cs["exp"], 1) < PMAP.get(row["Priority_Level"], 1):
        s -= abs(cs["w"])
    sat = int(row["Satisfaction_Score"])
    if sat <= 2 and row["Priority_Level"] in ("Low", "Medium"):
        s += 0.18
    elif sat >= 4 and row["Priority_Level"] in ("Critical", "High"):
        s -= 0.12
    rt = float(row["Resolution_Time_Hours"])
    exp = RT_BENCH.get((row["Issue_Category"], row["Priority_Level"]), 40.0)
    r = rt / max(exp, 1)
    if r < 0.4:
        s += 0.14
    elif r > 2.5:
        s += 0.10
    return s


def get_verdict(row, prob):
    base = PMAP.get(row["Priority_Level"], 1)
    d = dir_score(row)
    if d < 0:
        bump = 2 if prob >= 0.90 else 1
        mtype = "False Alarm"
        inf = max(0, base - bump)
    else:
        bump = 2 if prob >= 0.85 else 1
        mtype = "Hidden Crisis"
        inf = min(3, base + bump)
    return RPMAP[inf], mtype


def make_dossier(row, prob):
    inf, mtype = get_verdict(row, prob)
    t = f"{row['Ticket_Subject']} {row['Ticket_Description']}".lower()
    ev = []
    for pat, etype, w in ESCA_EV + DEESC_EV:
        m = re.search(pat, t)
        if m and (w > 0) == (mtype == "Hidden Crisis"):
            fld = "Ticket_Subject" if re.search(pat, row["Ticket_Subject"].lower()) else "Ticket_Description"
            ev.append({
                "signal": "keyword",
                "type": etype,
                "value": m.group(0),
                "source_field": fld,
                "weight": round(w, 3),
            })

    rt = float(row["Resolution_Time_Hours"])
    exp = RT_BENCH.get((row["Issue_Category"], row["Priority_Level"]), 40.0)
    r = rt / max(exp, 1)
    ev.append({
        "signal": "resolution_time",
        "value": f"{rt:.0f}h",
        "expected": f"~{exp:.0f}h",
        "source_field": "Resolution_Time_Hours",
        "weight": round(0.25 if r < 0.4 else 0.20 if r > 2.5 else 0.05, 3),
    })

    cs = CAT_SEV.get(row["Issue_Category"], {"exp": "Medium", "w": 0.05, "note": ""})
    ev.append({
        "signal": "category_baseline",
        "value": row["Issue_Category"],
        "source_field": "Issue_Category",
        "weight": round(abs(cs["w"]), 3),
    })

    ev.append({
        "signal": "satisfaction_score",
        "value": str(int(row["Satisfaction_Score"])),
        "source_field": "Satisfaction_Score",
        "weight": round(0.18 if int(row["Satisfaction_Score"]) <= 2 else 0.02, 3),
    })

    ev = sorted(ev, key=lambda x: abs(x.get("weight", 0)), reverse=True)

    a = PMAP.get(row["Priority_Level"], 1)
    ii = PMAP.get(inf, 1)
    d = ii - a
    delta = (
        f"+{d} (under-prioritised by {d} level{'s' if d > 1 else ''})"
        if d > 0
        else f"{d} (over-prioritised by {abs(d)} level{'s' if abs(d) > 1 else ''})"
        if d < 0
        else "0 (borderline)"
    )

    kp = [e for e in ev if e["signal"] == "keyword" and e.get("weight", 0) > 0]
    kn = [e for e in ev if e["signal"] == "keyword" and e.get("weight", 0) < 0]

    ch = row.get("Ticket_Channel", "Unknown")
    s1 = f"This {row['Issue_Category']} ticket via {ch} assigned {row['Priority_Level']} — model infers {inf}."
    s2 = (
        f"Escalation indicators ({', '.join(repr(e['value']) for e in kp[:2])}) signal higher severity."
        if mtype == "Hidden Crisis" and kp
        else f"Low-severity indicators suggest {row['Priority_Level']} is over-assigned."
        if mtype == "False Alarm" and kn
        else "Semantic patterns and metadata signal priority mismatch."
    )
    s3 = (
        f"RT={rt:.0f}h with sat={int(row['Satisfaction_Score'])}/5 supports under-prioritisation."
        if mtype == "Hidden Crisis"
        else f"RT={rt:.0f}h with sat={int(row['Satisfaction_Score'])}/5 consistent with over-triage."
    )

    return {
        "ticket_id": str(row.get("Ticket_ID", "")),
        "assigned_priority": row["Priority_Level"],
        "inferred_severity": inf,
        "mismatch_type": mtype,
        "severity_delta": delta,
        "confidence": round(float(prob), 4),
        "feature_evidence": ev,
        "constraint_analysis": f"{s1} {s2} {s3}",
    }


def zplot(fig, h=380):
    fig.update_layout(
        autosize=True,
        height=h,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def card_html(title, value, sub="", accent="orange"):
    accent_class = "pill" if accent == "orange" else "pill pill-blue"
    return f"""
    <div class="card">
        <div class="card-topline">
            <div class="{accent_class}">{title}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def rank_bars(series, color="orange", max_items=8):
    if series is None or len(series) == 0:
        st.info("No data available.")
        return
    s = series.sort_values(ascending=True).tail(max_items)
    maxv = float(s.max()) if float(s.max()) > 0 else 1.0
    for name, val in s.items():
        pct = float(val) / maxv * 100
        bar_class = "bar-fill" if color == "orange" else "bar-fill-blue"
        st.markdown(
            f"""
            <div style="margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;gap:0.75rem;margin-bottom:0.25rem;">
                    <div class="rank-name">{name}</div>
                    <div class="small-muted">{float(val):.1f}</div>
                </div>
                <div class="bar-wrap"><div class="{bar_class}" style="width:{pct:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def normalize_source_columns(df):
    if df is None or df.empty:
        return df

    rename_map = {}
    if "label" in df.columns and "predicted" not in df.columns:
        rename_map["label"] = "predicted"
    if "inferred_sev" in df.columns and "inferred_severity" not in df.columns:
        rename_map["inferred_sev"] = "inferred_severity"
    if "mismatch_type" in df.columns and "mtype" not in df.columns:
        rename_map["mismatch_type"] = "mtype"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "predicted" not in df.columns and "verdict" in df.columns:
        df["predicted"] = df["verdict"].map({"Consistent": 0, "Mismatch": 1})
    if "verdict" not in df.columns and "predicted" in df.columns:
        df["verdict"] = df["predicted"].map({0: "Consistent", 1: "Mismatch"})
    if "mtype" not in df.columns and "predicted" in df.columns:
        df["mtype"] = np.where(df["predicted"] == 1, "Mismatch", "Consistent")
    if "inferred_severity" not in df.columns and "mtype" in df.columns:
        df["inferred_severity"] = np.where(df["mtype"] == "Hidden Crisis", "Higher", "Lower")

    if "delta" not in df.columns and "severity_delta" in df.columns:
        df["delta"] = df["severity_delta"]

    return df


def load_batch_data():
    labeled_path = OUT / "labeled_tickets.csv"
    pred_path = OUT / "predictions.csv"
    dossiers_path = OUT / "evidence_dossiers.json"

    labeled = None
    preds = None
    dossiers = []

    if labeled_path.exists():
        labeled = normalize_source_columns(pd.read_csv(labeled_path))
    if pred_path.exists():
        preds = normalize_source_columns(pd.read_csv(pred_path))

    if dossiers_path.exists():
        try:
            with open(dossiers_path, "r", encoding="utf-8") as f:
                dossiers = json.load(f)
        except Exception:
            dossiers = []

    if labeled is None and preds is None:
        return None, dossiers, "none"

    if labeled is None:
        return preds, dossiers, "predictions"

    if preds is None:
        return labeled, dossiers, "labeled"

    # Merge predictions into labeled using Ticket_ID as the key.
    batch = labeled.copy()
    if "Ticket_ID" not in batch.columns or "Ticket_ID" not in preds.columns:
        return labeled, dossiers, "labeled"

    batch["Ticket_ID"] = batch["Ticket_ID"].astype(str)
    preds = preds.copy()
    preds["Ticket_ID"] = preds["Ticket_ID"].astype(str)

    pred_cols = [c for c in preds.columns if c != "Ticket_ID"]
    pred_map = preds.set_index("Ticket_ID")

    for col in pred_cols:
        mapped = batch["Ticket_ID"].map(pred_map[col].to_dict())
        if col in batch.columns:
            batch[col] = batch[col].where(batch[col].notna(), mapped)
        else:
            batch[col] = mapped

    batch = normalize_source_columns(batch)
    return batch, dossiers, "merged"


def top_value_safe(df, col):
    if df is None or df.empty or col not in df.columns:
        return "-"
    vc = df[col].dropna().astype(str).value_counts()
    return vc.idxmax() if len(vc) else "-"


# ---------------------------
# App
# ---------------------------
tok, model, THR = load_model()
model_ready = tok is not None

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;color:#ffeadf;">🔍 SIA — Support Integrity Auditor</h1>
        <p style="margin:0.35rem 0 0 0;color:#d4d9e3;">
            Detect priority mismatches in CRM support tickets · MARS Open Projects 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_ready:
    st.error("⚠️ Model not found at `models/sia_model/best`. Please ensure you have run the training pipeline first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "🚨 Intelligence Center", "📊 Executive Dashboard"])

# ---------------------------
# Tab 1 — Single Ticket
# ---------------------------
with tab1:
    st.subheader("Analyse a Single Ticket")
    c1, c2 = st.columns(2)

    with c1:
        subject = st.text_input("Ticket Subject", value="Cannot access my account")
        category = st.selectbox("Issue Category", ["Technical", "Fraud", "Account", "Billing", "General Inquiry"])
        channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Phone", "Web Form", "Social Media"])
        priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])

    with c2:
        desc = st.text_area(
            "Ticket Description",
            height=130,
            value="I have been locked out of my account since yesterday. I cannot access any of my data and payment is overdue.",
        )
        rt = st.number_input("Resolution Time (hours)", min_value=0.0, value=48.0, step=1.0)
        sat = st.slider("Satisfaction Score", 1, 5, 2)

    if st.button("🔍 Analyse Ticket", type="primary"):
        row = {
            "Ticket_ID": "SINGLE-001",
            "Ticket_Subject": subject,
            "Ticket_Description": desc,
            "Issue_Category": category,
            "Ticket_Channel": channel,
            "Priority_Level": priority,
            "Resolution_Time_Hours": rt,
            "Satisfaction_Score": sat,
        }
        txt = make_input(row)
        probs, preds = predict([txt], tok, model, THR)
        prob = probs[0]
        pred = preds[0]

        st.divider()
        col_a, col_b, col_c = st.columns(3)

        if pred == 1:
            inf, mtype = get_verdict(row, prob)
            col_a.metric("Verdict", "🚨 MISMATCH", mtype)
            col_b.metric("Confidence", f"{prob:.1%}")
            col_c.metric("Inferred Severity", inf, f"Assigned: {priority}")
            st.error(f"**{mtype}** — Assigned `{priority}` but model infers `{inf}`")
            dos = make_dossier(row, prob)
            with st.expander("📋 Evidence Dossier", expanded=True):
                st.json(dos)
        else:
            col_a.metric("Verdict", "✅ CONSISTENT")
            col_b.metric("Confidence (Mismatch)", f"{prob:.1%}")
            col_c.metric("Priority", priority)
            st.success(f"Priority **{priority}** appears correctly assigned. (confidence mismatch: {prob:.1%})")

# ---------------------------
# Tab 2 — Intelligence Center
# ---------------------------
with tab2:
    st.subheader("Intelligence Center")

    batch_df, dossiers, source_mode = load_batch_data()
    if batch_df is None:
        st.info("I could not find `outputs/labeled_tickets.csv` or `outputs/predictions.csv`. Run the pipeline first.")
    else:
        batch_df = normalize_source_columns(batch_df)

        flagged = batch_df[batch_df["predicted"] == 1].copy() if "predicted" in batch_df.columns else batch_df.iloc[0:0].copy()
        hidden = flagged[flagged["mtype"] == "Hidden Crisis"].copy() if "mtype" in flagged.columns else flagged.iloc[0:0].copy()
        falsea = flagged[flagged["mtype"] == "False Alarm"].copy() if "mtype" in flagged.columns else flagged.iloc[0:0].copy()

        top_cat = top_value_safe(flagged, "Issue_Category")
        top_ch = top_value_safe(flagged, "Ticket_Channel")
        avg_conf = float(flagged["prob"].mean()) * 100 if not flagged.empty and "prob" in flagged.columns else 0.0

        mode_label = {"merged": "Using both saved CSVs", "labeled": "Using labeled_tickets.csv", "predictions": "Using predictions.csv"}.get(source_mode, "Using saved outputs")
        st.caption(mode_label)

        st.markdown("### Executive Briefing")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(card_html("Hidden Crises", f"{len(hidden):,}", "Under-prioritised tickets", accent="orange"), unsafe_allow_html=True)
        with k2:
            st.markdown(card_html("False Alarms", f"{len(falsea):,}", "Over-prioritised tickets", accent="blue"), unsafe_allow_html=True)
        with k3:
            st.markdown(card_html("Flagged Total", f"{len(flagged):,}", f"of {len(batch_df):,} tickets", accent="orange"), unsafe_allow_html=True)
        with k4:
            st.markdown(card_html("Top Category", top_cat, "Highest mismatch concentration", accent="blue"), unsafe_allow_html=True)
        with k5:
            st.markdown(card_html("Avg Confidence", f"{avg_conf:.1f}%", "Mismatch probability", accent="orange"), unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="risk-title">Most Dangerous Hidden Crises</div>', unsafe_allow_html=True)
        if hidden.empty:
            st.info("No hidden crises available.")
        else:
            top_hidden = hidden.sort_values(["prob", "Ticket_ID"], ascending=[False, True]).head(6)
            cols = st.columns(2)
            for i, (_, r) in enumerate(top_hidden.iterrows()):
                with cols[i % 2]:
                    desc_txt = str(r["Ticket_Description"])[:220] if "Ticket_Description" in r.index else "Description unavailable in this CSV."
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-topline">
                                <div class="pill">Hidden Crisis</div>
                                <div class="pill pill-blue">{float(r.get('prob', 0))*100:.1f}%</div>
                            </div>
                            <div class="ticket-id">{r['Ticket_ID']}</div>
                            <div class="ticket-meta">
                                Assigned: <b>{r.get('Priority_Level', '—')}</b><br/>
                                Inferred: <b>{r.get('inferred_severity', r.get('inferred_sev', '—'))}</b><br/>
                                Category: <b>{r.get('Issue_Category', '—')}</b>
                                {"<br/>Channel: <b>" + str(r['Ticket_Channel']) + "</b>" if 'Ticket_Channel' in r.index else ""}
                            </div>
                            <div class="ticket-desc">{desc_txt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="risk-title">Resource Waste Alerts</div>', unsafe_allow_html=True)
        if falsea.empty:
            st.info("No false alarms available.")
        else:
            top_false = falsea.sort_values(["prob", "Ticket_ID"], ascending=[False, True]).head(6)
            cols = st.columns(2)
            for i, (_, r) in enumerate(top_false.iterrows()):
                with cols[i % 2]:
                    desc_txt = str(r["Ticket_Description"])[:220] if "Ticket_Description" in r.index else "Description unavailable in this CSV."
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-topline">
                                <div class="pill pill-blue">False Alarm</div>
                                <div class="pill">{float(r.get('prob', 0))*100:.1f}%</div>
                            </div>
                            <div class="ticket-id">{r['Ticket_ID']}</div>
                            <div class="ticket-meta">
                                Assigned: <b>{r.get('Priority_Level', '—')}</b><br/>
                                Inferred: <b>{r.get('inferred_severity', r.get('inferred_sev', '—'))}</b><br/>
                                Category: <b>{r.get('Issue_Category', '—')}</b>
                                {"<br/>Channel: <b>" + str(r['Ticket_Channel']) + "</b>" if 'Ticket_Channel' in r.index else ""}
                            </div>
                            <div class="ticket-desc">{desc_txt}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            st.markdown("### Risk Leaderboard")
            if not flagged.empty and "Issue_Category" in flagged.columns:
                cat_scores = flagged.groupby("Issue_Category")["predicted"].mean().sort_values() * 100
                rank_bars(cat_scores, color="orange", max_items=10)
            else:
                st.info("No category data available.")

        with c2:
            st.markdown("### Channel Risk Leaderboard")
            if not flagged.empty and "Ticket_Channel" in flagged.columns:
                ch_scores = flagged.groupby("Ticket_Channel")["predicted"].mean().sort_values() * 100
                rank_bars(ch_scores, color="blue", max_items=10)
            else:
                st.info("Channel data not present in this CSV.")

        st.markdown("### Escalation Hotspots")
        if flagged.empty or "Ticket_Channel" not in flagged.columns:
            st.info("Hotspot map needs Ticket_Channel data.")
        else:
            hotspot = (
                flagged.groupby(["Issue_Category", "Ticket_Channel"], dropna=False)["predicted"]
                .mean()
                .reset_index(name="Mismatch Rate")
            )
            hotspot["Mismatch Rate (%)"] = hotspot["Mismatch Rate"] * 100
            fig = px.density_heatmap(
                hotspot,
                x="Ticket_Channel",
                y="Issue_Category",
                z="Mismatch Rate (%)",
                color_continuous_scale="RdYlGn_r",
                title="Mismatch Rate by Category and Channel",
            )
            zplot(fig, h=520)

        st.markdown("### Investigation Queue")
        if flagged.empty:
            st.info("No tickets in the investigation queue.")
        else:
            queue_cols = [
                c for c in [
                    "Ticket_ID", "Priority_Level", "Issue_Category", "Ticket_Channel",
                    "prob", "mtype", "inferred_severity", "inferred_sev", "verdict"
                ] if c in flagged.columns
            ]
            qdf = flagged.sort_values("prob", ascending=False)[queue_cols].copy()
            if "prob" in qdf.columns:
                qdf["prob"] = (qdf["prob"] * 100).round(1).astype(str) + "%"
            st.dataframe(qdf, use_container_width=True, height=280)

            chosen_id = st.selectbox(
                "Open a ticket for full evidence",
                options=flagged["Ticket_ID"].astype(str).tolist(),
            )
            chosen_row = flagged[flagged["Ticket_ID"].astype(str) == str(chosen_id)].iloc[0].to_dict()
            chosen_prob = float(chosen_row.get("prob", 0.5))
            dossier = next((d for d in dossiers if str(d.get("ticket_id")) == str(chosen_id)), None)
            if dossier is None and {"Ticket_Subject", "Ticket_Description"}.issubset(chosen_row.keys()):
                dossier = make_dossier(chosen_row, chosen_prob)
            elif dossier is None:
                dossier = {"ticket_id": chosen_id, "note": "Evidence dossier not available for this row in the saved JSON."}

            with st.expander("Selected Ticket Evidence Dossier", expanded=True):
                st.json(dossier)

        st.markdown("### Executive Summary Table")
        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Tickets",
                    "Flagged Mismatches",
                    "Hidden Crises",
                    "False Alarms",
                    "Top Category",
                    "Top Channel",
                ],
                "Value": [
                    f"{len(batch_df):,}",
                    f"{len(flagged):,}",
                    f"{len(hidden):,}",
                    f"{len(falsea):,}",
                    top_cat,
                    top_ch if top_ch != "-" else "Not available",
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ---------------------------
# Tab 3 — Executive Dashboard
# ---------------------------
with tab3:
    st.subheader("Executive Dashboard")

    src = None
    if "batch_df" in st.session_state:
        src = st.session_state["batch_df"]
    else:
        src, _, _ = load_batch_data()
        if src is not None:
            src = normalize_source_columns(src)

    if src is None:
        st.info("Run training and batch prediction first to populate this dashboard.")
    else:
        src = normalize_source_columns(src)
        n_tot = len(src)
        n_mis = int(src["predicted"].sum()) if "predicted" in src.columns else 0
        hc = int(src[src.get("mtype", "") == "Hidden Crisis"].shape[0]) if "mtype" in src.columns else 0

        a, b, c = st.columns(3)
        a.markdown(card_html("Total Tickets", f"{n_tot:,}", "All analysed tickets", accent="blue"), unsafe_allow_html=True)
        b.markdown(card_html("Flagged Mismatches", f"{n_mis:,}", f"{(n_mis / n_tot * 100):.1f}%" if n_tot else "0%", accent="orange"), unsafe_allow_html=True)
        c.markdown(card_html("Hidden Crisis", f"{hc:,}", "Under-prioritised cases", accent="orange"), unsafe_allow_html=True)

        st.divider()

        left, right = st.columns(2)
        with left:
            st.markdown("### Mismatch Type Distribution")
            if "mtype" in src.columns:
                vc = src[src["mtype"].isin(["Hidden Crisis", "False Alarm", "Consistent"])]["mtype"].value_counts()
                if len(vc) > 0:
                    fig = px.bar(
                        x=vc.index,
                        y=vc.values,
                        color=vc.index,
                        color_discrete_map={
                            "Hidden Crisis": "#e05c2a",
                            "False Alarm": "#5b8fc9",
                            "Consistent": "#5e9e6e",
                        },
                        labels={"x": "", "y": "Count"},
                    )
                    fig.update_layout(autosize=True, height=420, showlegend=False)
                    zplot(fig, h=420)

        with right:
            st.markdown("### Mismatch Rate by Category")
            if "predicted" in src.columns and "Issue_Category" in src.columns:
                grp = src.groupby("Issue_Category")["predicted"].mean().sort_values(ascending=True) * 100
                fig = px.bar(
                    x=grp.values,
                    y=grp.index,
                    orientation="h",
                    color=grp.values,
                    color_continuous_scale="RdYlGn_r",
                    labels={"x": "Mismatch Rate (%)", "y": ""},
                )
                fig.update_layout(autosize=True, height=420, coloraxis_showscale=False)
                zplot(fig, h=420)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Severity Delta Heatmap")
            if "delta" in src.columns and "Ticket_Channel" in src.columns:
                pivot = src.pivot_table(
                    values="delta",
                    index="Issue_Category",
                    columns="Ticket_Channel",
                    aggfunc="mean",
                ).fillna(0)
                fig = px.imshow(
                    pivot,
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    text_auto=".2f",
                    aspect="auto",
                    labels={"color": "Mean Δ", "x": "Channel", "y": "Category"},
                )
                fig.update_layout(autosize=True, height=480)
                zplot(fig, h=480)
            elif "delta" in src.columns:
                st.info("Channel column unavailable for heatmap in this CSV.")

        with c2:
            st.markdown("### Evidence Signal Frequency")
            dos_path = OUT / "evidence_dossiers.json"
            if dos_path.exists():
                try:
                    with open(dos_path, "r", encoding="utf-8") as f:
                        dos_data = json.load(f)
                    sig_counts = {}
                    for d in dos_data:
                        for ev in d.get("feature_evidence", []):
                            sig = ev.get("type") or ev.get("signal", "unknown")
                            sig_counts[sig] = sig_counts.get(sig, 0) + 1
                    if sig_counts:
                        sc_df = pd.DataFrame(list(sig_counts.items()), columns=["Signal Type", "Count"])
                        sc_df = sc_df.sort_values("Count", ascending=True).tail(10)
                        fig = px.bar(
                            sc_df,
                            x="Count",
                            y="Signal Type",
                            orientation="h",
                            color="Count",
                            color_continuous_scale="Blues",
                        )
                        fig.update_layout(autosize=True, height=480, coloraxis_showscale=False)
                        zplot(fig, h=480)
                    else:
                        st.info("No signal data available yet.")
                except Exception:
                    st.info("Could not read evidence dossiers.")
            else:
                st.info("Run prediction first to generate signal evidence.")

        st.divider()
        st.markdown("### Category × Channel Risk Map")
        if "predicted" in src.columns and "Ticket_Channel" in src.columns:
            heat = (
                src.groupby(["Issue_Category", "Ticket_Channel"], dropna=False)["predicted"]
                .mean()
                .reset_index(name="Mismatch Rate")
            )
            if not heat.empty:
                heat["Mismatch Rate (%)"] = heat["Mismatch Rate"] * 100
                fig = px.density_heatmap(
                    heat,
                    x="Ticket_Channel",
                    y="Issue_Category",
                    z="Mismatch Rate (%)",
                    color_continuous_scale="RdYlGn_r",
                )
                zplot(fig, h=520)
        else:
            st.info("Category × Channel risk map needs Ticket_Channel data.")

st.divider()
st.caption("SIA · MARS Open Projects 2026 · DeBERTa-v3-small + LoRA · Intelligence-centered UI")
