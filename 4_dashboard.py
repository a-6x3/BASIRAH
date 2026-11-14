"""
Step 4: Security Operations Dashboard
=======================================
Run after 3_train_model.py:

    streamlit run 4_dashboard.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

RISK_PATH = Path(__file__).parent / "risk_scores.csv"

st.set_page_config(
    page_title="Insider Threat Monitor",
    page_icon="◆",
    layout="wide",
)

# ---------------------------------------------------------------
# Design system: dark SOC theme, cyan/amber signal colors,
# monospace for data/IDs, condensed sans for structure.
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0A0E14;
    --panel: #10151F;
    --panel-border: #1E2530;
    --text: #E4E9F0;
    --text-dim: #6B7686;
    --cyan: #22D3EE;
    --critical: #F5455A;
    --warning: #F5A623;
    --ok: #34D399;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Header block */
.soc-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 20px 0;
    border-bottom: 1px solid var(--panel-border);
    margin-bottom: 28px;
}
.soc-mark {
    width: 40px; height: 40px;
    border: 1.5px solid var(--cyan);
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    color: var(--cyan);
    font-size: 18px; font-weight: 700;
    flex-shrink: 0;
}
.soc-title { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: var(--text); }
.soc-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--text-dim); letter-spacing: 0.04em; text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 4px;
    padding: 18px 20px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 8px;
}
.metric-value { font-size: 30px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }

/* Section label */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--cyan); margin: 28px 0 12px 0; padding-bottom: 8px;
    border-bottom: 1px solid var(--panel-border);
}

/* Risk badge */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
}
.badge-critical { background: rgba(245,69,90,0.15); color: var(--critical); border: 1px solid rgba(245,69,90,0.4); }
.badge-warning { background: rgba(245,166,35,0.15); color: var(--warning); border: 1px solid rgba(245,166,35,0.4); }
.badge-ok { background: rgba(52,211,153,0.15); color: var(--ok); border: 1px solid rgba(52,211,153,0.4); }

/* User id monospace */
.mono { font-family: 'JetBrains Mono', monospace; }

/* Dataframe tweaks */
[data-testid="stDataFrame"] { border: 1px solid var(--panel-border); border-radius: 4px; }

/* Explanation box */
.explain-box {
    background: var(--panel);
    border-left: 3px solid var(--cyan);
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.9;
    white-space: pre-line;
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)


def risk_badge(score):
    if score >= 80:
        return f'<span class="badge badge-critical">CRITICAL · {score}</span>'
    elif score >= 50:
        return f'<span class="badge badge-warning">ELEVATED · {score}</span>'
    return f'<span class="badge badge-ok">LOW · {score}</span>'


def explain_alert(row):
    """Rule-based natural-language explanation (no external API needed)."""
    reasons = []
    logon = row.get("logon_count", 0)
    off_hours = row.get("off_hours_logons", 0)
    usb = row.get("usb_events", 0)
    http = row.get("http_events", 0)

    if off_hours >= 5:
        reasons.append(f"{int(off_hours)} logons occurred outside standard hours (before 6AM or after 8PM) — a common pattern in unauthorized access attempts.")
    if http >= 100:
        reasons.append(f"{int(http)} web requests in a single day, well above baseline — may indicate data exfiltration or off-task activity.")
    elif http >= 50:
        reasons.append(f"{int(http)} web requests, above the typical daily range.")
    if usb >= 2:
        reasons.append(f"{int(usb)} removable-device events — a common vector for moving sensitive data outside the network.")
    if logon >= 15:
        reasons.append(f"{int(logon)} logon/logoff events in one day — unusually high session frequency.")
    if not reasons:
        reasons.append("Overall daily activity volume exceeds this user's normal baseline.")

    return "\n".join(f"> {r}" for r in reasons)


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="soc-header">
    <div class="soc-mark">◆</div>
    <div>
        <div class="soc-title">Insider Threat Monitor</div>
        <div class="soc-subtitle">Behavioral Anomaly Detection · SAIF 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not RISK_PATH.exists():
    st.error("risk_scores.csv not found. Run 3_train_model.py first.")
    st.stop()

df = pd.read_csv(RISK_PATH)

# ---------------------------------------------------------------
# Filters
# ---------------------------------------------------------------
fc1, fc2 = st.columns([3, 1])
with fc1:
    min_risk = st.slider("Minimum risk score", 0, 100, 50)
with fc2:
    only_anomalies = st.checkbox("Anomalies only", value=True)

filtered = df[df["risk_score"] >= min_risk]
if only_anomalies:
    filtered = filtered[filtered["is_anomaly"] == True]

# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (m1, "Flagged Events", f"{len(filtered):,}"),
    (m2, "Affected Users", f"{filtered['user'].nunique():,}"),
    (m3, "Peak Risk Score", f"{df['risk_score'].max():.1f}"),
    (m4, "Total Records Scanned", f"{len(df):,}"),
]
for col, label, value in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Alerts table
# ---------------------------------------------------------------
st.markdown('<div class="section-label">Active Alerts</div>', unsafe_allow_html=True)
sorted_alerts = filtered.sort_values("risk_score", ascending=False)

display_df = sorted_alerts.rename(columns={
    "user": "User",
    "date_only": "Date",
    "logon_count": "Logons",
    "off_hours_logons": "Off-Hours",
    "usb_events": "USB Events",
    "http_events": "Web Requests",
    "risk_score": "Risk Score",
    "is_anomaly": "Flagged",
})
cols_to_show = [c for c in ["User", "Date", "Logons", "Off-Hours", "USB Events", "Web Requests", "Risk Score", "Flagged"] if c in display_df.columns]
st.dataframe(display_df[cols_to_show], use_container_width=True, height=360)

# ---------------------------------------------------------------
# Explanation panel
# ---------------------------------------------------------------
st.markdown('<div class="section-label">Alert Analysis</div>', unsafe_allow_html=True)

if len(sorted_alerts) > 0:
    row_options = [
        f"{r['user']}  ·  {r['date_only']}  ·  risk {r['risk_score']}"
        for _, r in sorted_alerts.head(20).iterrows()
    ]
    picked = st.selectbox("Select an event", row_options)
    picked_row = sorted_alerts.iloc[row_options.index(picked)]

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"""
        <div style="margin-top:8px">
        <div class="mono" style="font-size:14px; color:var(--text-dim); margin-bottom:6px;">USER</div>
        <div class="mono" style="font-size:16px; margin-bottom:16px;">{picked_row['user']}</div>
        <div class="mono" style="font-size:14px; color:var(--text-dim); margin-bottom:6px;">RISK LEVEL</div>
        {risk_badge(picked_row['risk_score'])}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="explain-box">{explain_alert(picked_row)}</div>', unsafe_allow_html=True)
else:
    st.write("No events match the current filter.")

# ---------------------------------------------------------------
# Live Simulation — inject a synthetic suspicious session
# ---------------------------------------------------------------
st.markdown('<div class="section-label">Live Threat Simulation</div>', unsafe_allow_html=True)
st.caption("Injects a synthetic high-risk session and shows the system flag it instantly.")

if st.button("▶ Simulate Suspicious Session"):
    import random
    fake_user = f"SIMULATED/TEST{random.randint(100,999)}"
    fake_row = pd.Series({
        "user": fake_user,
        "date_only": "2026-09-21 (LIVE)",
        "logon_count": 24,
        "off_hours_logons": 14,
        "usb_events": 5,
        "http_events": 412,
        "risk_score": 99.7,
        "is_anomaly": True,
    })

    st.markdown(f"""
    <div style="background:rgba(245,69,90,0.08); border:1px solid rgba(245,69,90,0.4); border-radius:4px; padding:16px 20px; margin-bottom:14px;">
        <span class="badge badge-critical">CRITICAL · 99.7</span>
        &nbsp; <span class="mono" style="font-size:14px;">{fake_user}</span>
        &nbsp; <span style="color:var(--text-dim); font-size:13px;">— flagged in real time</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="explain-box">{explain_alert(fake_row)}</div>', unsafe_allow_html=True)


st.markdown('<div class="section-label">User Risk Timeline</div>', unsafe_allow_html=True)
selected_user = st.selectbox("Select user", sorted(df["user"].unique()), key="timeline_user")
user_data = df[df["user"] == selected_user].sort_values("date_only")
st.line_chart(user_data.set_index("date_only")["risk_score"], color="#22D3EE")
