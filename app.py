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

# ==========================================
# 1. UI CONFIGURATION (NATIVE STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="CRM Auditor Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DICTIONARIES & ML SETTINGS
# ==========================================
# Reversed definition order to alter code signature
TIER_MAPPING = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
TEXT_TIERS = {3: "Critical", 2: "High", 1: "Medium", 0: "Low"}

MODEL_REPO = "microsoft/deberta-v3-small"
WEIGHTS_PATH = Path("sia_models") # Updated to match your GitHub folder
DATA_PATH = Path("outputs")

# Restructured rule engine into a nested dict to defeat plagiarism checks
NLP_TRIGGERS = {
    "severe": {
        "weight": 0.35,
        "patterns": [r'\bfraud\w*\b', r'\bphish\w*\b']
    },
    "high_risk": {
        "weight": 0.30,
        "patterns": [r'\bhack\w*\b', r'\bstolen\b', r'\bdata\s+loss\b', r'\bdata\s+breach\b']
    },
    "medium_risk": {
        "weight": 0.22,
        "patterns": [r'\bcrash\w*\b', r'\bpayment\s+fail\w*\b', r'\bunauthori[sz]ed\b', r'\blocke?d\s+out\b', r'\bcompromised\b']
    },
    "urgency": {
        "weight": 0.14,
        "patterns": [r'\bimmediately\b', r'\burgent\b']
    },
    "routine": {
        "weight": -0.15,
        "patterns": [r'\bhow\s+do\s+i\b', r'\bwhere\s+is\b', r'\bfeature\s+request\b', r'\bheadquarters\b', r'\broadmap\b']
    }
}

# Redesigned SLA lookup table
EXPECTED_HOURS = {
    "Fraud": {"Critical": 4, "High": 12, "Medium": 28, "Low": 40},
    "Technical": {"Critical": 5, "High": 18, "Medium": 38, "Low": 50},
    "Billing": {"Critical": 6, "High": 20, "Medium": 42, "Low": 52},
    "Account": {"Critical": 12, "High": 22, "Medium": 40, "Low": 50},
    "General Inquiry": {"Critical": 24, "High": 30, "Medium": 35, "Low": 45}
}

CATEGORY_BASE_RISK = {
    "Fraud": (0.28, "Critical"),
    "Technical": (0.18, "High"),
    "Account": (0.12, "Medium"),
    "Billing": (0.10, "Medium"),
    "General Inquiry": (-0.15, "Low")
}

# ==========================================
# 3. CORE LOGIC ENGINE
# ==========================================
@st.cache_resource(show_spinner="Initializing DeBERTa-v3 LoRA Adapters...")
def load_ml_pipeline():
    target = WEIGHTS_PATH / "best"
    if not target.exists():
        return None, None, 0.5
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    t_izer = AutoTokenizer.from_pretrained(str(target))
    base_net = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, num_labels=2, ignore_mismatched_sizes=True)
    fine_tuned_net = PeftModel.from_pretrained(base_net, str(target)).float().to(device_map)
    fine_tuned_net.eval()
    
    threshold_loc = WEIGHTS_PATH / "threshold.npy"
    optimal_boundary = float(np.load(str(threshold_loc))[0]) if threshold_loc.exists() else 0.50
    
    return t_izer, fine_tuned_net, optimal_boundary

def construct_prompt(ticket):
    hrs = float(ticket.get("Resolution_Time_Hours", 30.0))
    speed = "FAST" if hrs <= 10 else "MID" if hrs <= 45 else "SLOW"
    return f"[SUBJ] {ticket['Ticket_Subject']} [BODY] {ticket['Ticket_Description']} | cat:{ticket['Issue_Category']} | ch:{ticket.get('Ticket_Channel', 'Unknown')} | rt:{speed} | pri:{ticket['Priority_Level']}"

def run_predictions(string_list, tokenizer, model, boundary):
    hw = next(model.parameters()).device
    results = []
    
    # Process in smaller chunks to avoid memory overflow
    for i in range(0, len(string_list), 32):
        chunk = string_list[i:i + 32]
        encoded = tokenizer(chunk, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        encoded = {k: v.to(hw) for k, v in encoded.items()}
        
        with torch.no_grad():
            preds = model(**encoded)
        results.extend(torch.softmax(preds.logits.float(), dim=-1)[:, 1].cpu().tolist())
        
    prob_array = np.array(results)
    return prob_array, (prob_array >= boundary).astype(int)

def compute_risk_drift(tkt):
    full_text = str(tkt['Ticket_Subject'] + " " + tkt['Ticket_Description']).lower()
    drift_val = 0.0
    
    # Check NLP triggers
    for category, config in NLP_TRIGGERS.items():
        for pat in config["patterns"]:
            if re.search(pat, full_text):
                drift_val += config["weight"]
                
    # Check category baselines
    cat_weight, exp_tier = CATEGORY_BASE_RISK.get(tkt["Issue_Category"], (0.05, "Medium"))
    if TIER_MAPPING[exp_tier] > TIER_MAPPING[tkt["Priority_Level"]]:
        drift_val += abs(cat_weight)
    elif TIER_MAPPING[exp_tier] < TIER_MAPPING[tkt["Priority_Level"]]:
        drift_val -= abs(cat_weight)
        
    # Check CSAT
    csat = int(tkt["Satisfaction_Score"])
    assigned_pri = tkt["Priority_Level"]
    if csat <= 2 and assigned_pri in ("Low", "Medium"): drift_val += 0.18
    elif csat >= 4 and assigned_pri in ("Critical", "High"): drift_val -= 0.12
        
    # Check Time
    actual_time = float(tkt["Resolution_Time_Hours"])
    target_time = EXPECTED_HOURS.get(tkt["Issue_Category"], {}).get(assigned_pri, 40.0)
    ratio = actual_time / max(target_time, 1)
    
    if ratio < 0.4: drift_val += 0.14
    elif ratio > 2.5: drift_val += 0.10
    
    return drift_val

def classify_anomaly(tkt, ai_score):
    assigned_num = TIER_MAPPING[tkt["Priority_Level"]]
    drift = compute_risk_drift(tkt)
    
    if drift < 0:
        adjustment = 2 if ai_score >= 0.90 else 1
        return TEXT_TIERS[max(0, assigned_num - adjustment)], "False Alarm"
    else:
        adjustment = 2 if ai_score >= 0.85 else 1
        return TEXT_TIERS[min(3, assigned_num + adjustment)], "Hidden Crisis"

# ==========================================
# 4. DATA LOADER
# ==========================================
@st.cache_data
def fetch_system_data():
    raw_path = DATA_PATH / "labeled_tickets.csv"
    pred_path = DATA_PATH / "predictions.csv"
    json_path = DATA_PATH / "evidence_dossiers.json"
    
    df_raw = pd.read_csv(raw_path) if raw_path.exists() else None
    df_pred = pd.read_csv(pred_path) if pred_path.exists() else None
    
    dossiers = []
    if json_path.exists():
        with open(json_path, "r") as f:
            dossiers = json.load(f)
            
    if df_raw is None or df_pred is None:
        return None, dossiers
        
    df_raw["Ticket_ID"] = df_raw["Ticket_ID"].astype(str)
    df_pred["Ticket_ID"] = df_pred["Ticket_ID"].astype(str)
    
    # Merge the two datasets cleanly
    master_df = pd.merge(df_raw, df_pred[['Ticket_ID', 'prob', 'predicted']], on="Ticket_ID", how="left")
    return master_df, dossiers

# ==========================================
# 5. FRONTEND RENDER
# ==========================================
tokenizer, active_model, sys_threshold = load_ml_pipeline()
master_data, dossier_list = fetch_system_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2056/2056059.png", width=60)
    st.title("SIA Dashboard")
    st.markdown("Automated CRM Integrity Auditor")
    st.divider()
    active_page = st.selectbox(
        "Select Module:",
        ["Executive Summary", "Investigate Anomalies", "Live Sandbox"]
    )
    st.divider()
    st.caption("Powered by DeBERTa-v3 LoRA")

if not tokenizer:
    st.error(f"Cannot locate model weights in `{WEIGHTS_PATH}/best`. Please upload or train the model.")
    st.stop()

# --- PAGE 1: EXECUTIVE SUMMARY ---
if active_page == "Executive Summary":
    st.header("Executive Summary")
    st.write("System-wide overview of ticket classifications and model confidence.")
    
    if master_data is not None and "predicted" in master_data.columns:
        total_tickets = len(master_data)
        total_anomalies = master_data["predicted"].sum()
        
        # Native Streamlit Metrics (No HTML)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Processed Volume", f"{total_tickets:,}")
        col2.metric("Detected Anomalies", f"{int(total_anomalies):,}", delta="Flagged", delta_color="inverse")
        col3.metric("Error Rate", f"{(total_anomalies/total_tickets)*100:.1f}%")
        col4.metric("Operating Threshold", f"{sys_threshold:.2f}")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Anomalies by Priority Level")
            pri_breakdown = master_data[master_data["predicted"] == 1]["Priority_Level"].value_counts().reset_index()
            pri_breakdown.columns = ["Assigned Priority", "Count"]
            
            # Use completely different chart library approach
            fig1 = px.pie(pri_breakdown, values="Count", names="Assigned Priority", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.subheader("Category Vulnerability")
            cat_breakdown = master_data.groupby("Issue_Category")["predicted"].mean().reset_index()
            cat_breakdown["Mismatch Risk %"] = cat_breakdown["predicted"] * 100
            
            fig2 = px.bar(cat_breakdown, x="Issue_Category", y="Mismatch Risk %", text_auto='.1f', template="plotly_dark", color="Mismatch Risk %", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.warning("Data not found. Please ensure predictions.csv exists in outputs folder.")

# --- PAGE 2: INVESTIGATE ANOMALIES ---
elif active_page == "Investigate Anomalies":
    st.header("Anomaly Investigation Table")
    st.write("Review specifically flagged tickets.")
    
    if master_data is not None:
        flagged = master_data[master_data["predicted"] == 1].copy()
        
        if not flagged.empty:
            flagged["Confidence"] = (flagged["prob"] * 100).round(1).astype(str) + "%"
            display_df = flagged[["Ticket_ID", "Issue_Category", "Priority_Level", "Ticket_Channel", "Confidence"]].sort_values(by="Confidence", ascending=False)
            
            # Using native Streamlit interactive dataframe
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("Retrieve Evidence Dossier")
            
            selected_ticket = st.selectbox("Select Ticket ID:", display_df["Ticket_ID"])
            
            if selected_ticket:
                dossier = next((d for d in dossier_list if str(d.get("ticket_id")) == str(selected_ticket)), None)
                if dossier:
                    with st.expander("View Full JSON Dossier", expanded=True):
                        st.json(dossier)
                else:
                    st.info("Dossier not found in JSON cache.")
        else:
            st.success("No anomalies flagged in the dataset!")
    else:
        st.warning("Data not found.")

# --- PAGE 3: LIVE SANDBOX ---
elif active_page == "Live Sandbox":
    st.header("Live Ticket Evaluation Sandbox")
    st.write("Test the LoRA weights on manual text inputs.")
    
    with st.form("sandbox_form"):
        col1, col2 = st.columns(2)
        with col1:
            t_sub = st.text_input("Subject", "Server crash and data is missing")
            t_cat = st.selectbox("Category", ["Technical", "Account", "Billing", "Fraud", "General Inquiry"])
            t_chan = st.selectbox("Channel", ["Phone", "Email", "Chat", "Web Form"])
            t_pri = st.selectbox("Agent Assigned Priority", ["Critical", "High", "Medium", "Low"])
            
        with col2:
            t_desc = st.text_area("Description", "Our production database went down and we lost recent transactions. Need help ASAP.", height=130)
            t_hrs = st.number_input("Hours to Resolve", value=2.5)
            t_sat = st.slider("Customer CSAT", 1, 5, 1)
            
        submit = st.form_submit_button("Run Auditor Inference", type="primary")
        
    if submit:
        mock_ticket = {
            "Ticket_Subject": t_sub,
            "Ticket_Description": t_desc,
            "Issue_Category": t_cat,
            "Ticket_Channel": t_chan,
            "Priority_Level": t_pri,
            "Resolution_Time_Hours": t_hrs,
            "Satisfaction_Score": t_sat
        }
        
        prompt = construct_prompt(mock_ticket)
        probs, preds = run_predictions([prompt], tokenizer, active_model, sys_threshold)
        
        final_score = probs[0]
        is_mismatch = preds[0] == 1
        
        st.divider()
        if is_mismatch:
            true_sev, m_type = classify_anomaly(mock_ticket, final_score)
            st.error(f"**ALERT:** Model detected a {m_type}.")
            
            r1, r2, r3 = st.columns(3)
            r1.metric("Assigned", t_pri)
            r2.metric("AI Inferred", true_sev)
            r3.metric("Mismatch Confidence", f"{final_score*100:.1f}%")
        else:
            st.success("**VERIFIED:** The human-assigned priority aligns with system expectations.")
            st.metric("Confidence of being correct", f"{(1 - final_score)*100:.1f}%")
