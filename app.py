# app.py — Risk Management Portal (complete)
# Run: streamlit run app.py

import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- App setup ----------
st.set_page_config(
    page_title="Risk Management Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Risk Management Portal")
st.caption(
    "Define risks & controls, analyze exposure, and export reports. "
    "Use the sidebar to tune DIME weights globally."
)

# ---------- Sidebar: DIME Weights ----------
with st.sidebar:
    st.markdown("### DIME Weights")
    wD = st.slider("Design weight", 0.0, 1.0, 0.35, 0.05)
    wI = st.slider("Implementation weight", 0.0, 1.0, 0.35, 0.05)
    wM = st.slider("Monitoring weight", 0.0, 1.0, 0.15, 0.05)
    wE = st.slider("Evaluation weight", 0.0, 1.0, 0.15, 0.05)
    DIME_WEIGHTS = {"D": wD, "I": wI, "M": wM, "E": wE}

# ---------- Helper Functions ----------
def calculate_risk_score(likelihood, impact):
    if pd.isna(likelihood) or pd.isna(impact):
        return np.nan
    return likelihood * impact

def calculate_control_efficacy(design, implementation, monitoring, evaluation, w=None):
    """
    Weighted DIME efficacy on [0,1]. Returns np.nan if any score missing.
    Efficacy is 0 if Design or Implementation is 0.
    """
    if any(pd.isna(x) for x in [design, implementation, monitoring, evaluation]):
        return np.nan
    if design == 0 or implementation == 0:
        return 0.0
    if w is None:
        w = {"D": 0.35, "I": 0.35, "M": 0.15, "E": 0.15}
    total_w = w["D"] + w["I"] + w["M"] + w["E"]
    score = (
        w["D"] * design +
        w["I"] * implementation +
        w["M"] * monitoring +
        w["E"] * evaluation
    ) / (3 * total_w)
    return round(float(score), 4)

def calculate_residual_risk(inherent_likelihood, inherent_impact, control_eff_score, controlled_dimension):
    if pd.isna(inherent_likelihood) or pd.isna(inherent_impact) or pd.isna(control_eff_score):
        return np.nan
    adjusted_likelihood = inherent_likelihood
    adjusted_impact = inherent_impact
    if controlled_dimension == "Likelihood":
        adjusted_likelihood = max(1, round(inherent_likelihood * (1 - control_eff_score)))
    elif controlled_dimension == "Impact":
        adjusted_impact = max(1, round(inherent_impact * (1 - control_eff_score)))
    return adjusted_likelihood * adjusted_impact

def create_risk_matrix(df: pd.DataFrame):
    """Improved 5x5 matrix with grid + high-risk shading."""
    if df.empty:
        st.info("No data to chart.")
        return
    max_score = df["Residual Risk Score"].max() if df["Residual Risk Score"].notna().any() else 1
    fig = px.scatter(
        df,
        x="Likelihood",
        y="Impact",
        color="Residual Risk Score",
        size="Residual Risk Score",
        hover_name="Risk Name",
        hover_data={
            "Risk ID": True,
            "Inherent Risk Score": True,
            "Mapped Control": True,
            "Risk Category": True,
            "Status": True,
            "Residual Risk Score": True,
        },
        color_continuous_scale=px.colors.sequential.YlOrRd,
        range_color=[0, max_score],
    )
    shapes = []
    for i in range(1, 6):
        shapes.append(dict(type="line", x0=0.5, x1=5.5, y0=i, y1=i, line=dict(color="rgba(150,150,150,0.2)")))
        shapes.append(dict(type="line", x0=i, x1=i, y0=0.5, y1=5.5, line=dict(color="rgba(150,150,150,0.2)")))
    shapes.append(dict(type="rect", x0=3.5, x1=5.5, y0=3.5, y1=5.5, fillcolor="rgba(255,0,0,0.05)", line=dict(width=0)))
    fig.update_layout(
        title="Residual Risk Matrix",
        xaxis=dict(tickvals=[1,2,3,4,5], range=[0.5, 5.5], title="Likelihood"),
        yaxis=dict(tickvals=[1,2,3,4,5], range=[0.5, 5.5], title="Impact"),
        shapes=shapes,
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

def to_excel(df_dict: dict) -> bytes:
    """Create a multi-sheet Excel in-memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

def _read_table(file) -> pd.DataFrame | None:
    """Read CSV or Excel into DataFrame."""
    if file is None:
        return None
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def _normalize_risk_register_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types so st.data_editor column_config matches dataframe dtypes."""
    df = df.copy()
    if "Due Date" in df.columns:
        df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
    for col in ["Likelihood", "Impact", "Inherent Risk Score", "Residual Risk Score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "Risk ID", "Risk Name", "Description", "Mitigation Actions", "Risk Response",
        "Mapped Control", "Risk Category", "Period", "Owner", "Status"
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df

# --- JSON helpers (make DataFrames serializable) ---
def _clean_value(v):
    import numpy as _np
    import pandas as _pd
    if v is None:
        return None
    try:
        if _pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (_pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (_np.datetime64,)):
        return _pd.to_datetime(v).strftime("%Y-%m-%d")
    if isinstance(v, (_np.integer,)):
        return int(v)
    if isinstance(v, (_np.floating,)):
        return float(v)
    return v

def df_to_records_serializable(df: pd.DataFrame):
    records = []
    for rec in df.to_dict(orient="records"):
        records.append({k: _clean_value(v) for k, v in rec.items()})
    return records

# ---------- Session State Initialization ----------
if "risk_register" not in st.session_state:
    st.session_state.risk_register = pd.DataFrame({
        "Risk ID": ["R-001", "R-002"],
        "Risk Name": ["Cybersecurity Threat", "Supply Chain Disruption"],
        "Description": ["Unauthorized access to company data.", "Breakdown in critical supply chain."],
        "Likelihood": [4, 3],
        "Impact": [5, 4],
        "Inherent Risk Score": [20, 12],
        "Mitigation Actions": ["Implement stronger firewalls.", "Diversify key suppliers."],
        "Risk Response": ["Control", "Transfer"],
        "Mapped Control": ["Control A", "Control B"],
        "Risk Category": ["Operational", "Strategic"],
        "Period": ["Q1-2024", "Q1-2024"],
        "Owner": ["CISO", "COO"],
        "Status": ["Open", "In Progress"],
        "Due Date": ["2024-12-31", "2024-11-15"],
        "Residual Risk Score": [np.nan, np.nan],
    })

if "control_register" not in st.session_state:
    st.session_state.control_register = pd.DataFrame({
        "Control Name": ["Control A", "Control B"],
        "Description": ["Firewall policy.", "Supplier diversification."],
        "Dimension Controlled": ["Likelihood", "Impact"],
        "Design": [3, 2],
        "Implementation": [3, 3],
        "Monitoring": [3, 2],
        "Evaluation": [2, 3],
        "Control Efficacy Score": [0.83, 0.75],  # raw, recalculated each render
    })

# ---------- UI Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Risk Register", "Control Register", "Risk Analysis", "Reporting"])

# =========================
# Risk Register Tab
# =========================
with tab1:
    st.subheader("Risk Register")
    st.caption("Import from CSV/Excel or edit inline. Columns should at least include: Risk Name, Likelihood, Impact, Risk Category, Period.")

    # Import Risk Register here
    cL, cR = st.columns([1, 3])
    with cL:
        uploaded_rr = st.file_uploader("Import Risk Register (.csv/.xlsx)", type=["csv", "xlsx"], key="rr_import")
        if uploaded_rr:
            df_in = _read_table(uploaded_rr)
            if isinstance(df_in, pd.DataFrame):
                st.session_state.risk_register = df_in
                st.success("Risk Register imported.")
    with cR:
        st.info("Tip: You can also import/export in the Reporting tab.")

    control_list = ["None"] + st.session_state.control_register["Control Name"].dropna().unique().tolist()

    # Normalize types for editor
    df_rr = _normalize_risk_register_types(st.session_state.risk_register)

    edited_df = st.data_editor(
        df_rr,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Risk ID": st.column_config.TextColumn("Risk ID", disabled=True, help="Stable identifier"),
            "Risk Name": st.column_config.TextColumn("Risk Name", required=True),
            "Description": st.column_config.TextColumn("Description"),
            "Likelihood": st.column_config.NumberColumn("Likelihood (1-5)", min_value=1, max_value=5, step=1, required=True),
            "Impact": st.column_config.NumberColumn("Impact (1-5)", min_value=1, max_value=5, step=1, required=True),
            "Inherent Risk Score": st.column_config.NumberColumn("Inherent Risk Score", disabled=True, help="Likelihood × Impact"),
            "Mitigation Actions": st.column_config.TextColumn("Mitigation Actions"),
            "Risk Response": st.column_config.SelectboxColumn("Risk Response", options=["Acceptance", "Control", "Avoidance", "Transfer"], required=True),
            "Mapped Control": st.column_config.SelectboxColumn("Mapped Control", options=control_list, required=True),
            "Risk Category": st.column_config.SelectboxColumn("Risk Category", options=["Operational", "Financial", "Strategic", "Compliance"], required=True),
            "Period": st.column_config.TextColumn("Period", help="e.g., Q1-2024", required=True),
            "Owner": st.column_config.TextColumn("Owner", help="Accountable person/role", required=True),
            "Status": st.column_config.SelectboxColumn("Status", options=["Open", "In Progress", "Mitigated", "Closed"], required=True),
            "Due Date": st.column_config.DateColumn("Due Date"),
            "Residual Risk Score": st.column_config.NumberColumn("Residual Risk Score", disabled=True, help="Post-control score"),
        },
    )

    # Assign Risk IDs to new rows
    if "Risk ID" not in edited_df.columns:
        edited_df["Risk ID"] = np.nan
    missing_ids = edited_df["Risk ID"].isna()
    if missing_ids.any():
        existing_nums = pd.to_numeric(edited_df["Risk ID"].str[2:], errors="coerce")
        next_num = int(np.nanmax(existing_nums)) + 1 if existing_nums.notna().any() else 1
        new_ids = [f"R-{num:03d}" for num in range(next_num, next_num + missing_ids.sum())]
        edited_df.loc[missing_ids, "Risk ID"] = new_ids

    # Recalculate scores
    edited_df["Inherent Risk Score"] = edited_df.apply(
        lambda r: calculate_risk_score(r["Likelihood"], r["Impact"]), axis=1
    )
    control_map = (
        st.session_state.control_register.set_index("Control Name")
        if not st.session_state.control_register.empty
        else pd.DataFrame().set_index(pd.Index([]))
    )
    def _residual(row):
        name = row.get("Mapped Control", None)
        if name and name != "None" and name in control_map.index:
            crow = control_map.loc[name]
            eff = crow.get("Control Efficacy Score", np.nan)
            dim = crow.get("Dimension Controlled", None)
            return calculate_residual_risk(row["Likelihood"], row["Impact"], eff, dim)
        return row["Inherent Risk Score"]
    edited_df["Residual Risk Score"] = edited_df.apply(_residual, axis=1)

    st.session_state.risk_register = edited_df

# =========================
# Control Register Tab
# =========================
with tab2:
    st.subheader("Control Register")
    st.caption("Assess control efficacy using the weighted DIME framework (see sidebar).")

    controls_df = st.data_editor(
        st.session_state.control_register,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Control Name": st.column_config.TextColumn("Control Name", required=True),
            "Description": st.column_config.TextColumn("Description"),
            "Dimension Controlled": st.column_config.SelectboxColumn("Dimension Controlled", options=["Likelihood", "Impact"], required=True),
            "Design": st.column_config.NumberColumn("Design (0–3)", min_value=0, max_value=3, step=1),
            "Implementation": st.column_config.NumberColumn("Implementation (0–3)", min_value=0, max_value=3, step=1),
            "Monitoring": st.column_config.NumberColumn("Monitoring (0–3)", min_value=0, max_value=3, step=1),
            "Evaluation": st.column_config.NumberColumn("Evaluation (0–3)", min_value=0, max_value=3, step=1),
            "Control Efficacy Score": st.column_config.NumberColumn("Control Efficacy (raw)", disabled=True, help="0–1 value used in calculations"),
        },
    )
    controls_df["Control Efficacy Score"] = controls_df.apply(
        lambda r: calculate_control_efficacy(r["Design"], r["Implementation"], r["Monitoring"], r["Evaluation"], DIME_WEIGHTS),
        axis=1,
    )
    st.session_state.control_register = controls_df

# =========================
# Risk Analysis Tab
# =========================
with tab3:
    st.subheader("Risk Analysis")
    st.caption("Filter, aggregate, and visualize residual risk.")

    df = st.session_state.risk_register.copy()
    if df.empty or df["Risk Name"].isnull().all():
        st.info("No risks to display. Please add or import risks in the 'Risk Register' tab.")
        st.stop()

    st.markdown("#### Filter Risks")
    c1, c2, c3 = st.columns(3)
    risk_names = ["All"] + df["Risk Name"].dropna().unique().tolist()
    selected_risks = c1.multiselect("By Risk Name", options=risk_names, default=["All"])
    risk_categories = ["All"] + df["Risk Category"].dropna().unique().tolist()
    selected_categories = c2.multiselect("By Category", options=risk_categories, default=["All"])
    risk_periods = ["All"] + df["Period"].dropna().unique().tolist()
    selected_periods = c3.multiselect("By Period", options=risk_periods, default=["All"])

    filtered_df = df.copy()
    if "All" not in selected_risks:
        filtered_df = filtered_df[filtered_df["Risk Name"].isin(selected_risks)]
    if "All" not in selected_categories:
        filtered_df = filtered_df[filtered_df["Risk Category"].isin(selected_categories)]
    if "All" not in selected_periods:
        filtered_df = filtered_df[filtered_df["Period"].isin(selected_periods)]
    if filtered_df.empty:
        st.warning("No risks match the selected filters.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Risks (filtered)", len(filtered_df))
    k2.metric("Total Residual Score", f"{filtered_df['Residual Risk Score'].sum():.0f}")
    k3.metric("Open Risks", int((filtered_df["Status"] == "Open").sum()) if "Status" in filtered_df.columns else 0)

    st.markdown("#### Filtered Risk Register")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Aggregated Risk Score")
    total_residual_score = filtered_df["Residual Risk Score"].sum()
    st.metric("Total Aggregated Residual Risk Score", value=f"{total_residual_score:.2f}")

    st.markdown("---")
    st.markdown("#### Aggregated Score Breakdown")
    breakdown_param = st.selectbox(
        "Group Aggregated Score by:",
        options=["Risk Category", "Risk Response", "Period", "Owner", "Status"],
    )
    if breakdown_param:
        aggregated_breakdown = filtered_df.groupby(breakdown_param)["Residual Risk Score"].sum().reset_index()
        st.dataframe(aggregated_breakdown, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Residual Risk Matrix")
    create_risk_matrix(filtered_df)

# =========================
# Reporting Tab
# =========================
with tab4:
    st.subheader("Reporting")
    st.caption("Import/export data, save/load projects, and download Excel reports.")

    st.markdown("#### Import Data")
    ic1, ic2 = st.columns(2)
    with ic1:
        up_risks = st.file_uploader("Upload Risk Register (.xlsx/.csv)", type=["xlsx", "csv"], key="up_risks_rep")
    with ic2:
        up_controls = st.file_uploader("Upload Control Register (.xlsx/.csv)", type=["xlsx", "csv"], key="up_controls_rep")

    changed = False
    if up_risks:
        rr = _read_table(up_risks)
        if rr is not None:
            st.session_state.risk_register = rr
            changed = True
    if up_controls:
        cr = _read_table(up_controls)
        if cr is not None:
            st.session_state.control_register = cr
            changed = True
    if changed:
        st.success("Imported successfully. Switch tabs to review.")

    # Save / Load Project (JSON) with safe serialization
    st.markdown("#### Save / Load Project")
    pj1, pj2 = st.columns(2)
    with pj1:
        payload = {
            "risks": df_to_records_serializable(st.session_state.risk_register),
            "controls": df_to_records_serializable(st.session_state.control_register),
        }
        proj_bytes = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
        st.download_button(
            "Download Project (.json)",
            data=proj_bytes.getvalue(),
            file_name="risk_project.json",
            mime="application/json",
        )
    with pj2:
        proj_up = st.file_uploader("Load Project (.json)", type=["json"], key="proj_json")
        if proj_up:
            data = json.load(proj_up)
            st.session_state.risk_register = pd.DataFrame(data.get("risks", []))
            st.session_state.control_register = pd.DataFrame(data.get("controls", []))
            st.success("Project loaded.")

    st.markdown("#### Download Excel Report")
    df_risks = st.session_state.risk_register.copy()
    df_controls = st.session_state.control_register.copy()
    if df_risks.empty and df_controls.empty:
        st.info("No data to report.")
    else:
        if "Residual Risk Score" in df_risks.columns:
            df_risks["Residual Risk Score"] = pd.to_numeric(df_risks["Residual Risk Score"], errors="coerce")
        df_ranked = df_risks.sort_values(by="Residual Risk Score", ascending=False).reset_index(drop=True)
        report_data = {
            "Risk Register": df_risks,
            "Control Register": df_controls,
            "Ranked Risks": df_ranked,
        }
        st.download_button(
            label="Download Full Risk Report (.xlsx)",
            data=to_excel(report_data),
            file_name="risk_management_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
