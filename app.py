# app.py — MinRisk Increment A (Config + Inherent Risk Register)
# Run: python3 -m streamlit run app.py
# Python 3.9+ compatible

import io
import json
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# -------------------------------
# App Setup
# -------------------------------
st.set_page_config(page_title="MinRisk — Inherent Register (Increment A)", layout="wide")

st.title("MinRisk — Inherent Risk Register (Increment A)")
st.caption("Config first. Then capture the inherent risk register with 1–6 scales, "
           "descriptor-driven Likelihood/Impact, and Risk Code generation (DIV-DEP-UNIT-NNN).")

# -------------------------------
# Defaults (seed data)
# -------------------------------
DEFAULT_ORGSTRUCTURE = pd.DataFrame({
    "Division Code": ["FIN", "FIN", "OPS"],
    "Division Name": ["Finance", "Finance", "Operations"],
    "Department Code": ["TRD", "TRD", "SCM"],
    "Department Name": ["Treasury Dealing", "Treasury Dealing", "Supplies"],
    "Unit Code": ["AP", "AR", "WH"],
    "Internal Unit Name": ["Accounts Payable", "Accounts Receivable", "Warehousing"],
    "Active": [True, True, True],
})

DEFAULT_RISKTAXONOMY = pd.DataFrame({
    "Main Category Code": ["OPR", "FIN", "STR", "CMP"],
    "Main Category Name": ["Operational", "Financial", "Strategic", "Compliance"],
    "Sub-Category Code": ["CYB", "LIQ", "SCM", "KYC"],
    "Sub-Category Name": ["Cybersecurity", "Liquidity", "Supply Chain", "KYC/AML"],
    "Category": ["Operational", "Financial", "Strategic", "Compliance"],
    "Active": [True, True, True, True],
})

DEFAULT_SCALES_L = pd.DataFrame({
    "Descriptor": [
        "Improbable/Remote",
        "Unlikely/Might happen",
        "Possible",
        "Good Chance",
        "Probable/Likely",
        "Definitely/Certain",
    ],
    "Score": [1, 2, 3, 4, 5, 6],
})

DEFAULT_SCALES_I = pd.DataFrame({
    "Descriptor": [
        "Minimal or Insignificant",
        "Slight or Minor",
        "Moderate",
        "High",
        "Very High",
        "Severe or Catastrophic",
    ],
    "Score": [1, 2, 3, 4, 5, 6],
})

DEFAULT_DIME_WEIGHTS = pd.DataFrame({
    "Design": [0.35], "Implementation": [0.35], "Monitoring": [0.15], "Evaluation": [0.15]
})

DEFAULT_APPETITE = pd.DataFrame({
    "Lower": [0, 6, 13, 22],
    "Upper": [6, 13, 22, 36],
    "Band": ["LOW", "MODEST", "MODERATE", "HIGH"],
    "Color": ["GREEN", "YELLOW", "AMBER", "RED"],
})

DEFAULT_COUNTERS = pd.DataFrame({
    "Unit Code": ["AP", "AR", "WH"],
    "Last Counter": [6, 12, 4],
})

DEFAULT_USERS = pd.DataFrame({"User Display Name": ["Risk Manager", "CISO", "COO"]})

# Expected Inherent Register columns (+ optional Period)
INHERENT_COLUMNS = [
    "Risk Code", "Division", "Departments", "Internal Unit",
    "Risk Main Category", "Risk Sub-Category",
    "There is a risk of", "as a result of", "which may lead to",
    "Inherent Risk Likelihood", "Inherent Risk Impact", "Inherent Risk Severity", "Period"
]

# -------------------------------
# Session State Initialization
# -------------------------------
def _init_state_df(key: str, df: pd.DataFrame):
    if key not in st.session_state:
        st.session_state[key] = df.copy()

_init_state_df("cfg_org", DEFAULT_ORGSTRUCTURE)
_init_state_df("cfg_tax", DEFAULT_RISKTAXONOMY)
_init_state_df("cfg_scale_L", DEFAULT_SCALES_L)
_init_state_df("cfg_scale_I", DEFAULT_SCALES_I)
_init_state_df("cfg_dime_w", DEFAULT_DIME_WEIGHTS)
_init_state_df("cfg_appetite", DEFAULT_APPETITE)
_init_state_df("cfg_counters", DEFAULT_COUNTERS)
_init_state_df("cfg_users", DEFAULT_USERS)

if "inherent_register" not in st.session_state:
    st.session_state.inherent_register = pd.DataFrame(columns=INHERENT_COLUMNS)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = pd.DataFrame(columns=["Timestamp", "User", "Action", "Entity", "Identifier", "Details"])

# -------------------------------
# Helpers
# -------------------------------
def log_action(user: str, action: str, entity: str, identifier: str, details: str = ""):
    entry = {
        "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "User": user or "Unknown",
        "Action": action,
        "Entity": entity,
        "Identifier": identifier,
        "Details": details,
    }
    st.session_state.audit_log = pd.concat([st.session_state.audit_log, pd.DataFrame([entry])], ignore_index=True)

def normalize_inherent_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strings
    for col in [
        "Risk Code", "Division", "Departments", "Internal Unit",
        "Risk Main Category", "Risk Sub-Category",
        "There is a risk of", "as a result of", "which may lead to", "Period"
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    # descriptors kept as strings
    for col in ["Inherent Risk Likelihood", "Inherent Risk Impact"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    # numeric
    if "Inherent Risk Severity" in df.columns:
        df["Inherent Risk Severity"] = pd.to_numeric(df["Inherent Risk Severity"], errors="coerce")
    return df

def ensure_inherent_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in INHERENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[INHERENT_COLUMNS]

def descriptor_to_score(map_df: pd.DataFrame, descriptor: Optional[str]) -> Optional[int]:
    if descriptor is None or pd.isna(descriptor):
        return None
    row = map_df.loc[map_df["Descriptor"] == descriptor]
    if row.empty:
        return None
    return int(row["Score"].values[0])

def compute_inherent_severity(df: pd.DataFrame, scale_L: pd.DataFrame, scale_I: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores_L = df["Inherent Risk Likelihood"].apply(lambda d: descriptor_to_score(scale_L, d))
    scores_I = df["Inherent Risk Impact"].apply(lambda d: descriptor_to_score(scale_I, d))
    df["Inherent Risk Severity"] = pd.to_numeric(scores_L) * pd.to_numeric(scores_I)
    return df

def parse_risk_code_get_counter(code: str) -> Optional[int]:
    # Expect pattern DIV-DEP-UNIT-NNN
    try:
        if not isinstance(code, str) or "-" not in code:
            return None
        tail = code.split("-")[-1]
        return int(tail)
    except Exception:
        return None

def next_counter_for_unit(unit_code: str, existing_codes: List[str], counters_df: pd.DataFrame) -> int:
    # derive from existing codes first
    existing = [
        parse_risk_code_get_counter(code)
        for code in existing_codes
        if isinstance(code, str) and code.endswith(tuple([f"-{i:03d}" for i in range(1, 20000)]))
        and code.split("-")[-2] == unit_code
    ]
    max_existing = max([c for c in existing if c is not None], default=0)

    # seed from cfg_counters if higher
    seed = 0
    row = counters_df.loc[counters_df["Unit Code"] == unit_code]
    if not row.empty:
        try:
            seed = int(row["Last Counter"].values[0])
        except Exception:
            seed = 0

    return max(max_existing, seed) + 1

def compose_risk_code(div_code: str, dep_code: str, unit_code: str, counter: int) -> str:
    return f"{div_code}-{dep_code}-{unit_code}-{counter:03d}"

def update_counters(unit_code: str, new_counter: int):
    df = st.session_state.cfg_counters.copy()
    if unit_code in df["Unit Code"].values:
        st.session_state.cfg_counters.loc[df["Unit Code"] == unit_code, "Last Counter"] = new_counter
    else:
        st.session_state.cfg_counters = pd.concat([
            df,
            pd.DataFrame([{"Unit Code": unit_code, "Last Counter": new_counter}])
        ], ignore_index=True)

def generate_missing_risk_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows missing Risk Code, generate DIV-DEP-UNIT-NNN by looking up codes from OrgStructure
    and using a per-Unit counter (highest of existing codes or cfg_counters + 1).
    """
    df = df.copy()
    org = st.session_state.cfg_org
    # Build quick sets for validation (codes only)
    div_codes = set(org["Division Code"].dropna().astype(str))
    dep_codes = set(org["Department Code"].dropna().astype(str))
    unit_codes = set(org["Unit Code"].dropna().astype(str))

    existing_codes = df["Risk Code"].dropna().astype(str).tolist()

    for idx, row in df.iterrows():
        rc = row.get("Risk Code")
        if isinstance(rc, str) and rc.strip():
            continue  # already present
        div = str(row.get("Division") or "").strip()
        dep = str(row.get("Departments") or "").strip()
        unit = str(row.get("Internal Unit") or "").strip()
        # Validate presence in org codes
        if not (div in div_codes and dep in dep_codes and unit in unit_codes):
            # cannot generate without valid codes
            continue
        nxt = next_counter_for_unit(unit, existing_codes, st.session_state.cfg_counters)
        code = compose_risk_code(div, dep, unit, nxt)
        df.at[idx, "Risk Code"] = code
        existing_codes.append(code)
        update_counters(unit, nxt)
    return df

# ---- JSON serialization helpers (safe) ----
def _clean_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (pd.Timestamp, )):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (np.datetime64, )):
        return pd.to_datetime(v).strftime("%Y-%m-%d")
    if isinstance(v, (np.integer, )):
        return int(v)
    if isinstance(v, (np.floating, )):
        return float(v)
    return v

def df_records_serializable(df: pd.DataFrame):
    return [{k: _clean_value(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]

def save_project_bytes() -> bytes:
    payload = {
        "config": {
            "OrgStructure": df_records_serializable(st.session_state.cfg_org),
            "RiskTaxonomy": df_records_serializable(st.session_state.cfg_tax),
            "Scales_Likelihood": df_records_serializable(st.session_state.cfg_scale_L),
            "Scales_Impact": df_records_serializable(st.session_state.cfg_scale_I),
            "DIME_Weights": df_records_serializable(st.session_state.cfg_dime_w),
            "AppetiteBands": df_records_serializable(st.session_state.cfg_appetite),
            "Counters": df_records_serializable(st.session_state.cfg_counters),
            "Users": df_records_serializable(st.session_state.cfg_users),
        },
        "data": {
            "InherentRegister": df_records_serializable(st.session_state.inherent_register),
        },
        "audit": df_records_serializable(st.session_state.audit_log),
        "version": "A.1",
    }
    return json.dumps(payload, indent=2).encode("utf-8")

def load_project(file_bytes: bytes):
    payload = json.loads(file_bytes.decode("utf-8"))
    cfg = payload.get("config", {})
    data = payload.get("data", {})
    audit = payload.get("audit", [])

    st.session_state.cfg_org = pd.DataFrame(cfg.get("OrgStructure", []))
    st.session_state.cfg_tax = pd.DataFrame(cfg.get("RiskTaxonomy", []))
    st.session_state.cfg_scale_L = pd.DataFrame(cfg.get("Scales_Likelihood", []))
    st.session_state.cfg_scale_I = pd.DataFrame(cfg.get("Scales_Impact", []))
    st.session_state.cfg_dime_w = pd.DataFrame(cfg.get("DIME_Weights", []))
    st.session_state.cfg_appetite = pd.DataFrame(cfg.get("AppetiteBands", []))
    st.session_state.cfg_counters = pd.DataFrame(cfg.get("Counters", []))
    st.session_state.cfg_users = pd.DataFrame(cfg.get("Users", []))
    st.session_state.inherent_register = ensure_inherent_columns(pd.DataFrame(data.get("InherentRegister", [])))
    st.session_state.audit_log = pd.DataFrame(audit)

# -------------------------------
# Sidebar — Current user & Project I/O
# -------------------------------
with st.sidebar:
    st.markdown("### Session")
    current_user = st.selectbox(
        "Current user",
        options=st.session_state.cfg_users["User Display Name"].tolist(),
        index=0
    )
    st.caption("Used for audit log entries.")

    st.markdown("---")
    st.markdown("### Project Save/Load")
    b = st.download_button("Download Project (.json)", data=save_project_bytes(),
                           file_name="minrisk_project.json", mime="application/json")
    proj_up = st.file_uploader("Load Project (.json)", type=["json"], key="proj_json_all")
    if proj_up:
        load_project(proj_up.read())
        st.success("Project loaded.")
        log_action(current_user, "Load Project", "Project", "minrisk_project.json")

# -------------------------------
# Tabs
# -------------------------------
tab_cfg, tab_reg, tab_audit = st.tabs(["Config", "Inherent Register", "Audit Log"])

# ===============================
# Config Tab
# ===============================
with tab_cfg:
    st.subheader("Configuration")
    st.caption("Edit your reference data. All lists drive dropdowns and validations in the register. "
               "DIME weights/Appetite are global and will apply in later increments.")

    sub1, sub2 = st.tabs(["Organization & Taxonomy", "Scales, Appetite & DIME"])

    with sub1:
        st.markdown("#### Organization Structure")
        st.info("Codes must be unique and **Active=True** to appear in dropdowns.")
        st.session_state.cfg_org = st.data_editor(
            st.session_state.cfg_org,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Division Code": st.column_config.TextColumn("Division Code", required=True),
                "Division Name": st.column_config.TextColumn("Division Name"),
                "Department Code": st.column_config.TextColumn("Department Code", required=True),
                "Department Name": st.column_config.TextColumn("Department Name"),
                "Unit Code": st.column_config.TextColumn("Unit Code", required=True),
                "Internal Unit Name": st.column_config.TextColumn("Internal Unit Name"),
                "Active": st.column_config.CheckboxColumn("Active", default=True),
            },
        )

        st.markdown("#### Risk Taxonomy")
        st.session_state.cfg_tax = st.data_editor(
            st.session_state.cfg_tax,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Main Category Code": st.column_config.TextColumn("Main Category Code", required=True),
                "Main Category Name": st.column_config.TextColumn("Main Category Name"),
                "Sub-Category Code": st.column_config.TextColumn("Sub-Category Code", required=True),
                "Sub-Category Name": st.column_config.TextColumn("Sub-Category Name"),
                "Category": st.column_config.SelectboxColumn(
                    "Category", options=["Operational","Financial","Strategic","Compliance"], required=True),
                "Active": st.column_config.CheckboxColumn("Active", default=True),
            },
        )

        st.markdown("#### Counters (per Unit Code)")
        st.session_state.cfg_counters = st.data_editor(
            st.session_state.cfg_counters,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Unit Code": st.column_config.TextColumn("Unit Code", required=True),
                "Last Counter": st.column_config.NumberColumn("Last Counter", min_value=0, step=1),
            },
        )

        st.markdown("#### Users (for Audit)")
        st.session_state.cfg_users = st.data_editor(
            st.session_state.cfg_users,
            num_rows="dynamic", use_container_width=True,
            column_config={"User Display Name": st.column_config.TextColumn("User Display Name", required=True)},
        )

    with sub2:
        st.markdown("#### Likelihood Scale (Descriptors → Scores 1–6)")
        st.session_state.cfg_scale_L = st.data_editor(
            st.session_state.cfg_scale_L,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Descriptor": st.column_config.TextColumn("Descriptor", required=True),
                "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=6, step=1, required=True),
            },
        )

        st.markdown("#### Impact Scale (Descriptors → Scores 1–6)")
        st.session_state.cfg_scale_I = st.data_editor(
            st.session_state.cfg_scale_I,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Descriptor": st.column_config.TextColumn("Descriptor", required=True),
                "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=6, step=1, required=True),
            },
        )

        st.markdown("#### Appetite Bands (Editable)")
        st.session_state.cfg_appetite = st.data_editor(
            st.session_state.cfg_appetite,
            num_rows="dynamic", use_container_width=True,
            column_config={
                "Lower": st.column_config.NumberColumn("Lower", min_value=0, max_value=36, step=1),
                "Upper": st.column_config.NumberColumn("Upper", min_value=1, max_value=36, step=1),
                "Band": st.column_config.SelectboxColumn(
                    "Band", options=["LOW","MODEST","MODERATE","HIGH"], required=True),
                "Color": st.column_config.SelectboxColumn(
                    "Color", options=["GREEN","YELLOW","AMBER","RED"], required=True),
            },
        )

        st.markdown("#### DIME Weights (Global)")
        st.session_state.cfg_dime_w = st.data_editor(
            st.session_state.cfg_dime_w,
            num_rows="fixed", use_container_width=True,
            column_config={
                "Design": st.column_config.NumberColumn("Design", min_value=0.0, max_value=1.0, step=0.01),
                "Implementation": st.column_config.NumberColumn("Implementation", min_value=0.0, max_value=1.0, step=0.01),
                "Monitoring": st.column_config.NumberColumn("Monitoring", min_value=0.0, max_value=1.0, step=0.01),
                "Evaluation": st.column_config.NumberColumn("Evaluation", min_value=0.0, max_value=1.0, step=0.01),
            },
        )
        st.caption("Weights are normalized in computation; zero-override applies later when we add DIME efficacy.")

# ===============================
# Inherent Register Tab
# ===============================
with tab_reg:
    st.subheader("Inherent Risk Register")
    st.caption("Use descriptor pickers for Likelihood/Impact. Risk Code is generated from Org codes + per-Unit counter.")

    # Import CSV
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Import Inherent Register (.csv)", type=["csv"], key="inherent_csv")
        if up:
            df_in = pd.read_csv(up)
            df_in = ensure_inherent_columns(df_in)
            st.session_state.inherent_register = df_in
            st.success("Inherent Register imported.")
            log_action(current_user, "Import", "Inherent Register", up.name, f"rows={len(df_in)}")
    with c2:
        out_csv = st.session_state.inherent_register.to_csv(index=False).encode("utf-8")
        st.download_button("Download Current Register (CSV)", data=out_csv,
                           file_name="inherent_register.csv", mime="text/csv")

    # Dropdown options from Config
    org_active = st.session_state.cfg_org[st.session_state.cfg_org["Active"] == True]
    div_opts = sorted(org_active["Division Code"].dropna().astype(str).unique().tolist())
    dep_opts = sorted(org_active["Department Code"].dropna().astype(str).unique().tolist())
    unit_opts = sorted(org_active["Unit Code"].dropna().astype(str).unique().tolist())

    tax_active = st.session_state.cfg_tax[st.session_state.cfg_tax["Active"] == True]
    main_cat_opts = sorted(tax_active["Main Category Code"].dropna().astype(str).unique().tolist())
    sub_cat_opts = sorted(tax_active["Sub-Category Code"].dropna().astype(str).unique().tolist())

    L_opts = st.session_state.cfg_scale_L["Descriptor"].dropna().astype(str).tolist()
    I_opts = st.session_state.cfg_scale_I["Descriptor"].dropna().astype(str).tolist()

    # Normalize types, compute severity for display
    reg_df = normalize_inherent_types(st.session_state.inherent_register)
    reg_df = ensure_inherent_columns(reg_df)
    reg_df = compute_inherent_severity(reg_df, st.session_state.cfg_scale_L, st.session_state.cfg_scale_I)

    edited = st.data_editor(
        reg_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Risk Code": st.column_config.TextColumn("Risk Code", disabled=True, help="Generated: DIV-DEP-UNIT-NNN"),
            "Division": st.column_config.SelectboxColumn("Division (Code)", options=div_opts, required=True),
            "Departments": st.column_config.SelectboxColumn("Departments (Code)", options=dep_opts, required=True),
            "Internal Unit": st.column_config.SelectboxColumn("Internal Unit (Code)", options=unit_opts, required=True),
            "Risk Main Category": st.column_config.SelectboxColumn("Risk Main Category (Code)", options=main_cat_opts, required=True),
            "Risk Sub-Category": st.column_config.SelectboxColumn("Risk Sub-Category (Code)", options=sub_cat_opts, required=True),
            "There is a risk of": st.column_config.TextColumn("There is a risk of", required=True),
            "as a result of": st.column_config.TextColumn("as a result of", required=True),
            "which may lead to": st.column_config.TextColumn("which may lead to", required=True),
            "Inherent Risk Likelihood": st.column_config.SelectboxColumn("Inherent Risk Likelihood (Descriptor)", options=L_opts, required=True),
            "Inherent Risk Impact": st.column_config.SelectboxColumn("Inherent Risk Impact (Descriptor)", options=I_opts, required=True),
            "Inherent Risk Severity": st.column_config.NumberColumn("Inherent Risk Severity", disabled=True, help="Score = L×I"),
            "Period": st.column_config.TextColumn("Period (e.g., Q1-2025)", help="Quarter label; optional"),
        },
    )

    # Generate missing Risk Codes (based on codes in row)
    edited = generate_missing_risk_codes(edited)
    # Recompute severity again (if user changed descriptors)
    edited = compute_inherent_severity(edited, st.session_state.cfg_scale_L, st.session_state.cfg_scale_I)
    # Persist
    st.session_state.inherent_register = edited

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Risks", len(edited))
    if "Inherent Risk Severity" in edited.columns and len(edited) > 0:
        k2.metric("Avg Severity", f"{pd.to_numeric(edited['Inherent Risk Severity'], errors='coerce').mean():.1f}")
        k3.metric("Max Severity", f"{pd.to_numeric(edited['Inherent Risk Severity'], errors='coerce').max():.0f}")
    else:
        k2.metric("Avg Severity", "—"); k3.metric("Max Severity", "—")

    # Buttons for audit logging
    cols = st.columns(3)
    if cols[0].button("Log: Saved Inherent Register"):
        log_action(current_user, "Save", "Inherent Register", "current", f"rows={len(st.session_state.inherent_register)}")
        st.success("Saved (logged).")

# ===============================
# Audit Log Tab
# ===============================
with tab_audit:
    st.subheader("Audit Log")
    st.dataframe(st.session_state.audit_log, use_container_width=True)
    if st.session_state.audit_log.empty:
        st.info("No audit entries yet.")
