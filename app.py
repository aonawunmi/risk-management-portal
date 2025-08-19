# app.py -- A Streamlit-based Risk Management Portal
# Run with: streamlit run app.py

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io

# ---------- App setup ----------
st.set_page_config(
    page_title="Risk Management Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Risk Management Portal")
st.markdown(
    """
    This portal allows you to manage and analyze your risks using a comprehensive framework.
    Start with the **Risk Register** to define your risks and controls.
    """
)

# ---------- Helper Functions ----------
def calculate_risk_score(likelihood, impact):
    if pd.isna(likelihood) or pd.isna(impact):
        return np.nan
    return likelihood * impact

def calculate_control_efficacy(design, implementation, monitoring, evaluation):
    if pd.isna(design) or pd.isna(implementation) or pd.isna(monitoring) or pd.isna(evaluation):
        return np.nan
    if design == 0 or implementation == 0:
        return 0
    return (design + implementation + monitoring + evaluation) / 12

def calculate_residual_risk(inherent_likelihood, inherent_impact, control_efficacy_score, controlled_dimension):
    if pd.isna(inherent_likelihood) or pd.isna(inherent_impact) or pd.isna(control_efficacy_score):
        return np.nan
    
    adjusted_likelihood = inherent_likelihood
    adjusted_impact = inherent_impact
    
    if controlled_dimension == 'Likelihood':
        adjusted_likelihood = round(inherent_likelihood * (1 - control_efficacy_score))
    elif controlled_dimension == 'Impact':
        adjusted_impact = round(inherent_impact * (1 - control_efficacy_score))
        
    return adjusted_likelihood * adjusted_impact

def create_risk_matrix(df: pd.DataFrame):
    max_score = df['Residual Risk Score'].max() if not df.empty else 1
    colors = px.colors.sequential.YlOrRd
    
    fig = px.scatter(
        df,
        x='Likelihood',
        y='Impact',
        color='Residual Risk Score',
        size='Residual Risk Score',
        color_continuous_scale=colors,
        range_color=[0, max_score],
        labels={
            "Likelihood": "Likelihood (1-5)",
            "Impact": "Impact (1-5)",
            "Residual Risk Score": "Residual Risk Score"
        },
        hover_data={'Risk Name': True, 'Inherent Risk Score': True, 'Mitigation Actions': True}
    )

    fig.update_layout(
        title={'text': "Residual Risk Matrix", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5], title="Likelihood"),
        yaxis=dict(tickvals=[1, 2, 3, 4, 5], range=[0.5, 5.5], title="Impact"),
        height=500,
        width=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def to_excel(df_dict: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


# ---------- Session State Initialization ----------
if "risk_register" not in st.session_state:
    st.session_state.risk_register = pd.DataFrame({
        "Risk Name": ["Cybersecurity Threat", "Supply Chain Disruption"],
        "Description": ["Unauthorized access to company data.", "Breakdown in critical supply chain." ],
        "Likelihood": [4, 3],
        "Impact": [5, 4],
        "Inherent Risk Score": [20, 12],
        "Mitigation Actions": ["Implement stronger firewalls.", "Diversify key suppliers."],
        "Risk Response": ["Control", "Transfer"],
        "Mapped Control": ["Control A", "Control B"],
        "Risk Category": ["Operational", "Strategic"],
        "Period": ["Q1-2024", "Q1-2024"]
    })

if "control_register" not in st.session_state:
    st.session_state.control_register = pd.DataFrame({
        "Control Name": ["Control A", "Control B"],
        "Description": ["Firewall policy.", "Supplier diversification." ],
        "Dimension Controlled": ["Likelihood", "Impact"],
        "Design": [3, 2],
        "Implementation": [3, 3],
        "Monitoring": [3, 2],
        "Evaluation": [2, 3],
        "Control Efficacy Score": [0.83, 0.75]
    })


# ---------- UI Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Risk Register", "Control Register", "Risk Analysis", "Reporting"])

# =========================
# Risk Register Tab
# =========================
with tab1:
    st.subheader("Risk Register")
    st.caption("Define and manage all identified risks.")

    control_list = ['None'] + st.session_state.control_register['Control Name'].unique().tolist()
    
    edited_df = st.data_editor(
        st.session_state.risk_register,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Likelihood": st.column_config.NumberColumn("Likelihood (1-5)", min_value=1, max_value=5, step=1, required=True),
            "Impact": st.column_config.NumberColumn("Impact (1-5)", min_value=1, max_value=5, step=1, required=True),
            "Inherent Risk Score": st.column_config.NumberColumn("Inherent Risk Score", disabled=True),
            "Risk Response": st.column_config.SelectboxColumn("Risk Response", options=["Acceptance", "Control", "Avoidance", "Transfer"], required=True),
            "Mapped Control": st.column_config.SelectboxColumn("Mapped Control", options=control_list, required=True),
            "Risk Category": st.column_config.SelectboxColumn("Risk Category", options=["Operational", "Financial", "Strategic", "Compliance"], required=True),
            "Period": st.column_config.TextColumn("Period", required=True),
            "Residual Risk Score": st.column_config.NumberColumn("Residual Risk Score", disabled=True)
        }
    )

    edited_df['Inherent Risk Score'] = edited_df.apply(lambda row: calculate_risk_score(row['Likelihood'], row['Impact']), axis=1)
    
    control_map = st.session_state.control_register.set_index('Control Name')
    def get_residual_risk(row):
        control_name = row['Mapped Control']
        if control_name and control_name != 'None' and control_name in control_map.index:
            control_row = control_map.loc[control_name]
            efficacy = control_row['Control Efficacy Score']
            dimension = control_row['Dimension Controlled']
            return calculate_residual_risk(row['Likelihood'], row['Impact'], efficacy, dimension)
        return row['Inherent Risk Score']
        
    edited_df['Residual Risk Score'] = edited_df.apply(get_residual_risk, axis=1)
    st.session_state.risk_register = edited_df

# =========================
# Control Register Tab
# =========================
with tab2:
    st.subheader("Control Register")
    st.caption("List and assess the efficacy of your risk controls using the DIME framework.")

    edited_df_controls = st.data_editor(
        st.session_state.control_register,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Dimension Controlled": st.column_config.SelectboxColumn("Dimension Controlled", options=["Likelihood", "Impact"], required=True),
            "Design": st.column_config.NumberColumn("Design (0-3)", min_value=0, max_value=3, step=1),
            "Implementation": st.column_config.NumberColumn("Implementation (0-3)", min_value=0, max_value=3, step=1),
            "Monitoring": st.column_config.NumberColumn("Monitoring (0-3)", min_value=0, max_value=3, step=1),
            "Evaluation": st.column_config.NumberColumn("Evaluation (0-3)", min_value=0, max_value=3, step=1),
            "Control Efficacy Score": st.column_config.NumberColumn("Control Efficacy Score", disabled=True)
        }
    )

    edited_df_controls['Control Efficacy Score'] = edited_df_controls.apply(
        lambda row: calculate_control_efficacy(row['Design'], row['Implementation'], row['Monitoring'], row['Evaluation']), axis=1
    )

    st.session_state.control_register = edited_df_controls

# =========================
# Risk Analysis Tab
# =========================
with tab3:
    st.subheader("Risk Analysis")
    st.caption("Analyze your risks by filtering and aggregating data.")
    
    df = st.session_state.risk_register.copy()
    if df.empty or df['Risk Name'].isnull().all():
        st.info("No risks to display. Please add risks in the 'Risk Register' tab.")
        st.stop()
    
    st.markdown("#### Filter Risks")
    filter_cols = st.columns(3)
    
    risk_names = ['All'] + df['Risk Name'].unique().tolist()
    selected_risks = filter_cols[0].multiselect("Filter by Risk Name", options=risk_names, default=['All'])
    
    risk_categories = ['All'] + df['Risk Category'].unique().tolist()
    selected_categories = filter_cols[1].multiselect("Filter by Risk Category", options=risk_categories, default=['All'])
    
    risk_periods = ['All'] + df['Period'].unique().tolist()
    selected_periods = filter_cols[2].multiselect("Filter by Period", options=risk_periods, default=['All'])
    
    filtered_df = df.copy()
    if 'All' not in selected_risks:
        filtered_df = filtered_df[filtered_df['Risk Name'].isin(selected_risks)]
    if 'All' not in selected_categories:
        filtered_df = filtered_df[filtered_df['Risk Category'].isin(selected_categories)]
    if 'All' not in selected_periods:
        filtered_df = filtered_df[filtered_df['Period'].isin(selected_periods)]

    if filtered_df.empty:
        st.warning("No risks match the selected filters.")
        st.stop()
    
    st.markdown("#### Filtered Risk Register")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Aggregated Risk Score")
    total_residual_score = filtered_df['Residual Risk Score'].sum()
    st.metric("Total Aggregated Residual Risk Score", value=f"{total_residual_score:.2f}")

    st.markdown("---")
    st.markdown("#### Aggregated Score Breakdown")
    breakdown_param = st.selectbox("Group Aggregated Score by:", options=['Risk Category', 'Risk Response', 'Period'])
    if breakdown_param:
        aggregated_breakdown = filtered_df.groupby(breakdown_param)['Residual Risk Score'].sum().reset_index()
        st.dataframe(aggregated_breakdown, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Residual Risk Matrix")
    create_risk_matrix(filtered_df)

# =========================
# Reporting Tab
# =========================
with tab4:
    st.subheader("Reporting")
    st.caption("Download your full risk data as an Excel file.")

    df_risks = st.session_state.risk_register
    df_controls = st.session_state.control_register

    if df_risks.empty and df_controls.empty:
        st.info("No data to report. Please add risks in the 'Risk Register' tab.")
    else:
        st.markdown("#### Download Report")
        df_ranked_risks = df_risks.sort_values(by="Residual Risk Score", ascending=False).reset_index(drop=True)
        report_data = {"Risk Register": df_risks, "Control Register": df_controls, "Ranked Risks": df_ranked_risks}
        st.download_button(label="Download Full Risk Report (.xlsx)", data=to_excel(report_data), file_name="risk_management_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
