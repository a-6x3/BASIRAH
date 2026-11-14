"""
InsiderGuard: judge-ready insider-threat monitor.

    streamlit run app.py

Data/model artifacts live in ./data (built by scripts/build_artifacts.py).
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import engine as E
from i18n import T

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FEATS = E.FEATURES
SCEN_ORDER = ["after_hours", "usb_copy", "web_exfil"]
LVL_ORDER = ["subtle", "moderate", "strong"]

# Quick scenarios applied on top of the chosen employee's normal day
# order of deltas: logons, off-hours logons, USB events, web requests
PRESETS = {
    "normal": [0, 0, 0, 0],
    "night": [8, 8, 0, 0],
    "usb": [2, 2, 6, 30],
    "web": [0, 0, 0, 250],
}
SLIDER_MAX = {"logon_count": 40, "off_hours_logons": 20, "usb_events": 10, "http_events": 700}
SKEYS = {"logon_count": "s_logon", "off_hours_logons": "s_off", "usb_events": "s_usb", "http_events": "s_http"}

st.set_page_config(page_title="InsiderGuard", page_icon="◆", layout="wide")

# --------------------------------------------------------------------------
# language
# --------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"


def t(key: str, **kw) -> str:
    s = T[st.session_state.lang].get(key, T["en"].get(key, key))
    return s.format(**kw) if kw else s


RTL = st.session_state.lang == "ar"

# --------------------------------------------------------------------------
# design system (dark SOC theme, from the original dashboard)
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
:root{--bg:#0A0E14;--panel:#10151F;--panel-border:#1E2530;--text:#E4E9F0;--text-dim:#8A94A6;
--cyan:#22D3EE;--critical:#F5455A;--warning:#F5A623;--ok:#34D399;}
html, body, [class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg);color:var(--text);}
#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1180px;}
.soc-header{display:flex;align-items:center;gap:14px;padding:4px 0 16px 0;border-bottom:1px solid var(--panel-border);margin-bottom:14px;}
.soc-mark{width:40px;height:40px;border:1.5px solid var(--cyan);display:flex;align-items:center;justify-content:center;
font-family:'JetBrains Mono',monospace;color:var(--cyan);font-size:18px;font-weight:700;flex-shrink:0;}
.soc-title{font-size:22px;font-weight:800;letter-spacing:-0.02em;color:var(--text);}
.soc-subtitle{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-dim);letter-spacing:.04em;text-transform:uppercase;}
.demo-banner{background:rgba(245,166,35,.08);border:1px solid rgba(245,166,35,.35);color:#F5C56B;border-radius:4px;
padding:8px 14px;font-size:13px;margin-bottom:16px;}
.metric-card{background:var(--panel);border:1px solid var(--panel-border);border-radius:4px;padding:16px 18px;}
.metric-label{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-dim);margin-bottom:8px;}
.metric-value{font-size:28px;font-weight:800;font-family:'JetBrains Mono',monospace;}
.section-label{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;
color:var(--cyan);margin:26px 0 12px 0;padding-bottom:8px;border-bottom:1px solid var(--panel-border);}
.badge{display:inline-block;padding:2px 10px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;}
.badge-critical{background:rgba(245,69,90,.15);color:var(--critical);border:1px solid rgba(245,69,90,.4);}
.badge-warning{background:rgba(245,166,35,.15);color:var(--warning);border:1px solid rgba(245,166,35,.4);}
.badge-ok{background:rgba(52,211,153,.15);color:var(--ok);border:1px solid rgba(52,211,153,.4);}
.mono{font-family:'JetBrains Mono',monospace;}
.dim{color:var(--text-dim);}
.card{background:var(--panel);border:1px solid var(--panel-border);border-radius:4px;padding:16px 20px;margin-bottom:12px;}
.explain-box{background:var(--panel);border-left:3px solid var(--cyan);border-radius:0 4px 4px 0;padding:14px 18px;
font-size:14px;line-height:1.8;color:var(--text);}
.explain-box div{margin:2px 0;}
.gauge{position:relative;height:16px;border-radius:8px;background:linear-gradient(90deg,#34D399 0%,#F5A623 55%,#F5455A 85%);margin:10px 0 4px 0;}
.gauge-dot{position:absolute;top:-5px;width:6px;height:26px;background:#fff;border-radius:3px;box-shadow:0 0 0 2px #0A0E14;}
.gauge-scale{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);}
.status{border-radius:4px;padding:12px 16px;font-weight:600;font-size:14px;margin-top:10px;}
.status-high{background:rgba(245,69,90,.1);border:1px solid rgba(245,69,90,.45);color:#FF8A98;}
.status-routine{background:rgba(245,166,35,.08);border:1px solid rgba(245,166,35,.35);color:#F5C56B;}
.status-none{background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.35);color:#6EE7B7;}
table.cmp{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px;border:0 !important;}
table.cmp td, table.cmp th{border-left:0 !important;border-right:0 !important;border-top:0 !important;background:transparent !important;}
table.cmp th{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.05em;text-transform:uppercase;
color:var(--text-dim);text-align:left;padding:6px 8px;border-bottom:1px solid var(--panel-border);font-weight:500;}
table.cmp td{padding:7px 8px;border-bottom:1px solid #151B26;font-family:'JetBrains Mono',monospace;}
table.cmp td.hl{color:#FF8A98;font-weight:700;}
.pts{display:inline-block;height:8px;border-radius:4px;background:var(--cyan);vertical-align:middle;margin-right:6px;}
.ext a{color:var(--cyan);}
.footer{color:var(--text-dim);font-size:12px;margin-top:36px;padding-top:14px;border-top:1px solid var(--panel-border);}
div[data-testid="stSlider"], [data-testid="stDataFrame"], .gauge, .gauge-scale, table.cmp{direction:ltr;}
</style>
""", unsafe_allow_html=True)

if RTL:
    st.markdown("""
<style>
html, body, [class*="css"], .stApp{font-family:'IBM Plex Sans Arabic','Inter',sans-serif;}
[data-testid="stMainBlockContainer"], .block-container{direction:rtl;text-align:right;}
.explain-box{border-left:none;border-right:3px solid var(--cyan);border-radius:4px 0 0 4px;}
.mono, table.cmp td, table.cmp th{direction:ltr;unicode-bidi:plaintext;}
table.cmp th{text-align:right;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# data + model (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data ...")
def load_scores() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "scores.parquet")
    df["user"] = df["user"].astype(str)
    df["role"] = df["role"].astype(str)
    df["priority"] = df["priority"].astype(str).replace("nan", "")
    return df


@st.cache_resource(show_spinner="Preparing the model ...")
def load_model():
    scores = pd.read_parquet(DATA / "scores.parquet", columns=FEATS)
    bundle, _ = E.load_or_train(DATA / "model.joblib", scores)
    return bundle


@st.cache_data
def load_json(name: str):
    return E.load_json(DATA / name)


df = load_scores()
bundle = load_model()
personas = load_json("personas.json")
cases = load_json("cases.json")
evaluation = load_json("eval.json")
astats = evaluation["alert_stats"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fmt_num(v: float) -> str:
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def risk_badge(score: float) -> str:
    tr = E.tier(score)
    cls = {"CRITICAL": "badge-critical", "ELEVATED": "badge-warning", "LOW": "badge-ok"}[tr]
    return f'<span class="badge {cls}">{t("tier_" + tr)} · {score:.1f}</span>'


def gauge(score: float) -> str:
    return (f'<div class="gauge"><div class="gauge-dot" style="left:calc({score:.1f}% - 3px)"></div></div>'
            f'<div class="gauge-scale"><span>0</span><span>50</span><span>80</span><span>100</span></div>')


def status_box(prio: str) -> str:
    if prio == "HIGH":
        return f'<div class="status status-high">{t("status_high")}</div>'
    if prio == "ROUTINE":
        return f'<div class="status status-routine">{t("status_routine")}</div>'
    return f'<div class="status status-none">{t("status_none")}</div>'


def analyse(x, med, p95):
    """Score one day and explain it. x, med, p95: length-4 arrays (med/p95 may hold NaN)."""
    x = np.asarray(x, float)
    med = np.asarray(med, float)
    p95 = np.asarray(p95, float)
    risk, flagged = E.score_matrix(bundle, x)
    risk, flagged = float(risk[0]), bool(flagged[0])
    has_base = not (np.isnan(med).any() or np.isnan(p95).any())
    neutral = med if has_base else np.zeros(len(x))
    contrib = E.contributions(bundle, x, neutral)
    unusual = E.unusual_flags(x, p95) if has_base else np.zeros(len(x), bool)
    prio = E.priority(flagged, bool(unusual.any()), has_base)
    usual_risk = float(E.score_matrix(bundle, med)[0][0]) if has_base else None
    return {"risk": risk, "flagged": flagged, "contrib": contrib, "unusual": unusual,
            "prio": prio, "has_base": has_base, "x": x, "med": med, "p95": p95,
            "usual_risk": usual_risk}


def reasons_html(a) -> str:
    items = []
    if not a["has_base"]:
        items.append(t("r_nobase"))
    order = np.argsort(-a["contrib"])
    for i in order:
        f = FEATS[i]
        if a["unusual"][i] or (a["flagged"] and a["contrib"][i] >= 8 and a["has_base"]):
            items.append(t("r_" + f, v=fmt_num(a["x"][i]),
                           m=fmt_num(a["med"][i]) if a["has_base"] else "?",
                           p=fmt_num(a["p95"][i]) if a["has_base"] else "?"))
    if not items:
        items.append(t("r_none"))
    if a["flagged"] and a["usual_risk"] is not None and a["usual_risk"] >= 50:
        items.append(t("r_usual", u=f"{a['usual_risk']:.0f}"))
    return "".join(f"<div>› {s}</div>" for s in items)


def compare_table(a, show_verdict: bool = True) -> str:
    """Today vs the user's own history. With show_verdict=False the highlights and
    the model's per-feature points are left out (the analyst decides first)."""
    rows = []
    for i, f in enumerate(FEATS):
        hl = " class='hl'" if (show_verdict and a["unusual"][i]) else ""
        cells = (f"<td style='font-family:inherit'>{t('f_' + f)}</td>"
                 f"<td{hl}>{fmt_num(a['x'][i])}</td>"
                 f"<td>{fmt_num(a['med'][i]) if a['has_base'] else '·'}</td>"
                 f"<td>{fmt_num(a['p95'][i]) if a['has_base'] else '·'}</td>")
        if show_verdict:
            pts = a["contrib"][i]
            cells += f'<td><span class="pts" style="width:{min(pts, 100) * 1.2:.0f}px"></span>{pts:.0f}</td>'
        rows.append(f"<tr>{cells}</tr>")
    head = (f"<tr><th></th><th>{t('col_today')}</th><th>{t('col_usual')}</th><th>{t('col_max')}</th>"
            + (f"<th>{t('col_pts')}</th>" if show_verdict else "") + "</tr>")
    return f"<table class='cmp'>{head}{''.join(rows)}</table>"


def kpi(col, label, value):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def chart(c):
    st.altair_chart(c, width="stretch")


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
hc1, hc2 = st.columns([6, 1])
with hc1:
    st.markdown(f"""<div class="soc-header"><div class="soc-mark">◆</div><div>
<div class="soc-title">{t('app_title')}</div><div class="soc-subtitle">{t('subtitle')}</div></div></div>""",
                unsafe_allow_html=True)
with hc2:
    st.radio("Language", ["en", "ar"], key="lang", horizontal=True, label_visibility="collapsed",
             format_func=lambda v: "EN" if v == "en" else "عربي")
st.markdown(f'<div class="demo-banner">{t("demo_banner")}</div>', unsafe_allow_html=True)

tab_exp, tab_soc, tab_ev, tab_rc = st.tabs([t("tab_exp"), t("tab_soc"), t("tab_ev"), t("tab_rc")])

# ==========================================================================
# TAB 1 — EXPERIENCE
# ==========================================================================
with tab_exp:
    st.markdown(f"### {t('exp_title')}")
    st.write(t("exp_intro"))

    # ---------------- Step 1: be the insider ----------------
    st.markdown(f'<div class="section-label">{t("step1_title")}</div>', unsafe_allow_html=True)
    st.caption(t("step1_help"))

    def persona_normal(p):
        return [int(round(v)) for v in p["median"]]

    def set_sliders(vals):
        for f, v in zip(FEATS, vals):
            st.session_state[SKEYS[f]] = int(min(max(v, 0), SLIDER_MAX[f]))

    def persona_label(i):
        return f"{t('persona_' + personas[i]['kind'])} · {personas[i]['role']} · {personas[i]['user'].split('/')[-1]}"

    def cur_persona():
        # robust across Streamlit versions: the stored value may be an index or its label
        v = st.session_state.get("persona_idx", 0)
        if isinstance(v, int):
            return v if 0 <= v < len(personas) else 0
        for i in range(len(personas)):
            if v == persona_label(i) or v == str(i):
                return i
        return 0

    def on_persona_change():
        set_sliders(persona_normal(personas[cur_persona()]))

    def apply_preset(name):
        base = persona_normal(personas[cur_persona()])
        set_sliders([b + d for b, d in zip(base, PRESETS[name])])

    if "persona_idx" not in st.session_state:
        st.session_state.persona_idx = 0
        on_persona_change()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.selectbox(t("pick_employee"), range(len(personas)), key="persona_idx",
                     on_change=on_persona_change,
                     format_func=persona_label)
        st.caption(t("preset_label"))
        pc = st.columns(2)
        for j, name in enumerate(["normal", "night", "usb", "web"]):
            pc[j % 2].button(t("preset_" + name), key=f"btn_{name}", on_click=apply_preset, args=(name,),
                             width="stretch")
        sliders = {}
        for f in FEATS:
            sliders[f] = st.slider(t("f_" + f), 0, SLIDER_MAX[f], key=SKEYS[f])

    P = personas[cur_persona()]
    a1 = analyse([sliders[f] for f in FEATS], P["median"], P["p95"])
    with right:
        st.markdown(f'<div class="metric-label">{t("risk_label")}</div>'
                    f'<div style="display:flex;align-items:center;gap:12px"><span class="metric-value">{a1["risk"]:.1f}</span>'
                    f'{risk_badge(a1["risk"])}</div>{gauge(a1["risk"])}{status_box(a1["prio"])}',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label" style="margin-top:16px">{t("why_title")}</div>'
                    f'<div class="explain-box">{reasons_html(a1)}</div>', unsafe_allow_html=True)
        st.markdown(compare_table(a1), unsafe_allow_html=True)

    # ---------------- Step 2: be the analyst ----------------
    st.markdown(f'<div class="section-label">{t("step2_title")}</div>', unsafe_allow_html=True)
    st.caption(t("step2_help"))

    ORDER = [i for i in [0, 2, 1, 3] if i < len(cases)]
    N = len(ORDER)
    st.session_state.setdefault("a_idx", 0)
    st.session_state.setdefault("a_answers", {})

    def answer(choice):
        st.session_state.a_answers[st.session_state.a_idx] = choice

    def next_case():
        st.session_state.a_idx += 1

    def restart():
        st.session_state.a_idx = 0
        st.session_state.a_answers = {}

    idx = st.session_state.a_idx
    if idx < N:
        c = cases[ORDER[idx]]
        ca = analyse(c["x"], c["median"], c["p95"])
        answered = idx in st.session_state.a_answers
        st.markdown(f"**{t('alert_n', n=idx + 1, t=N)}**")
        cl, cr = st.columns([1, 1], gap="large")
        with cl:
            st.markdown(f'<div class="card"><span class="dim">{t("case_user")}</span> '
                        f'<span class="mono">{c["user"]}</span><br><span class="dim">{t("case_role")}</span> {c["role"]}<br>'
                        f'<span class="dim">{t("case_risk")}</span> {risk_badge(c["risk"])}</div>', unsafe_allow_html=True)
            if not answered:
                b1, b2 = st.columns(2)
                b1.button(t("btn_investigate"), key=f"inv{idx}", on_click=answer, args=("investigate",), width="stretch")
                b2.button(t("btn_dismiss"), key=f"dis{idx}", on_click=answer, args=("dismiss",), width="stretch")
        with cr:
            # activity next to the user's own history; the system's verdict is hidden until you decide
            st.markdown(compare_table(ca, show_verdict=False), unsafe_allow_html=True)

        if answered:
            mine = st.session_state.a_answers[idx]
            truth_attack = c["truth"] == "attack"
            right_call = (mine == "investigate") == truth_attack
            sys_call = "investigate" if c["priority"] == "HIGH" else "dismiss"
            st.markdown(f'<div class="status {"status-none" if right_call else "status-high"}">'
                        f'{t("correct") if right_call else t("wrong")} · {t("your_call")}: '
                        f'{t("v_investigate") if mine == "investigate" else t("v_dismiss")} · '
                        f'{t("system_call")}: {t("v_investigate") if sys_call == "investigate" else t("v_dismiss")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="explain-box" style="margin-top:10px">'
                        f'<div>{t("reveal_attack") if truth_attack else t("reveal_routine")}</div>{reasons_html(ca)}</div>',
                        unsafe_allow_html=True)
            st.button(t("next_case") if idx + 1 < N else t("see_results"), key=f"nx{idx}", on_click=next_case)
    else:
        mine_ok = sum(
            ((st.session_state.a_answers.get(i) == "investigate") == (cases[ORDER[i]]["truth"] == "attack"))
            for i in range(N))
        naive_ok = sum(cases[ORDER[i]]["truth"] == "attack" for i in range(N))   # investigates every flagged day
        sys_ok = sum(((cases[ORDER[i]]["priority"] == "HIGH") == (cases[ORDER[i]]["truth"] == "attack"))
                     for i in range(N))
        st.markdown(f"**{t('res_title')}**")
        st.markdown(f'<div class="explain-box"><div>{t("res_you", a=mine_ok, t=N)}</div>'
                    f'<div>{t("res_naive", n=naive_ok, t=N)}</div><div>{t("res_system", s=sys_ok, t=N)}</div></div>',
                    unsafe_allow_html=True)
        st.caption(t("next_hint"))
        st.button(t("restart"), on_click=restart)

# ==========================================================================
# TAB 2 — SOC DASHBOARD
# ==========================================================================
with tab_soc:
    f1, f2 = st.columns([3, 2])
    with f1:
        min_risk = st.slider(t("soc_min_risk"), 0, 100, 50)
    with f2:
        view = st.radio(t("soc_show"), ["all", "high", "routine"], index=1, horizontal=True,
                        format_func=lambda v: t("prio_" + v))

    flagged_df = df[df["is_anomaly"] & (df["risk_score"] >= min_risk)]
    filt = flagged_df
    if view == "high":
        filt = flagged_df[flagged_df["priority"] == "HIGH"]
    elif view == "routine":
        filt = flagged_df[flagged_df["priority"] == "ROUTINE"]

    cols = st.columns(5)
    kpi(cols[0], t("m_flagged"), f"{len(flagged_df):,}")
    kpi(cols[1], t("m_high"), f"{(flagged_df['priority'] == 'HIGH').sum():,}")
    kpi(cols[2], t("m_routine"), f"{(flagged_df['priority'] == 'ROUTINE').sum():,}")
    kpi(cols[3], t("m_users"), f"{filt['user'].nunique():,}")
    kpi(cols[4], t("m_total"), f"{len(df):,}")

    st.markdown(f'<div class="section-label">{t("sec_alerts")}</div>', unsafe_allow_html=True)
    ranked = filt.sort_values("risk_score", ascending=False)
    show = ranked.rename(columns={
        "user": t("c_user"), "role": t("c_role"), "date_only": t("c_date"),
        "logon_count": t("f_logon_count"), "off_hours_logons": t("f_off_hours_logons"),
        "usb_events": t("f_usb_events"), "http_events": t("f_http_events"),
        "risk_score": t("c_risk"), "priority": t("c_prio"),
    })
    show_cols = [t("c_user"), t("c_role"), t("c_date"), t("f_logon_count"), t("f_off_hours_logons"),
                 t("f_usb_events"), t("f_http_events"), t("c_risk"), t("c_prio")]
    st.dataframe(show[show_cols].head(1000), width="stretch", height=360, hide_index=True)

    st.markdown(f'<div class="section-label">{t("sec_analysis")}</div>', unsafe_allow_html=True)
    if len(ranked) == 0:
        st.write(t("no_events"))
    else:
        top = ranked.head(25).reset_index(drop=True)
        pick = st.selectbox(t("select_event"), range(len(top)),
                            format_func=lambda i: f"{top.loc[i, 'user']} · {top.loc[i, 'date_only']} · risk {top.loc[i, 'risk_score']:.1f}")
        r = top.loc[pick]
        med = np.array([r[f"b_med_{f}"] for f in FEATS], float)
        p95 = np.array([r[f"b_p95_{f}"] for f in FEATS], float)
        ar = analyse([r[f] for f in FEATS], med, p95)
        d1, d2 = st.columns([1, 3])
        with d1:
            st.markdown(f"""<div style="margin-top:8px">
<div class="mono dim" style="font-size:12px;margin-bottom:4px">{t('lbl_user')}</div>
<div class="mono" style="font-size:16px;margin-bottom:12px">{r['user']}</div>
<div class="mono dim" style="font-size:12px;margin-bottom:4px">{t('lbl_role')}</div>
<div style="margin-bottom:12px">{r['role']}</div>
<div class="mono dim" style="font-size:12px;margin-bottom:6px">{t('lbl_level')}</div>{risk_badge(float(r['risk_score']))}</div>""",
                        unsafe_allow_html=True)
        with d2:
            st.markdown(status_box(ar["prio"]) + f'<div class="explain-box" style="margin-top:10px">{reasons_html(ar)}</div>',
                        unsafe_allow_html=True)
            st.markdown(compare_table(ar), unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{t("sec_timeline")}</div>', unsafe_allow_html=True)
    users = sorted(df["user"].unique())
    default_user = ranked["user"].iloc[0] if len(ranked) else users[0]
    sel_user = st.selectbox(t("select_user"), users, index=users.index(default_user))
    ud = df[df["user"] == sel_user].sort_values("date_only").copy()
    ud["date"] = pd.to_datetime(ud["date_only"])
    base = alt.Chart(ud).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(color="#22D3EE", strokeWidth=1.4).encode(y=alt.Y("risk_score:Q", title=t("c_risk"), scale=alt.Scale(domain=[0, 100])))
    pts = alt.Chart(ud[ud["priority"].isin(["HIGH", "ROUTINE"])]).mark_circle(size=55).encode(
        x="date:T", y="risk_score:Q",
        color=alt.Color("priority:N", scale=alt.Scale(domain=["HIGH", "ROUTINE"], range=["#F5455A", "#F5A623"]), legend=None),
        tooltip=["date_only", "risk_score", "priority"])
    rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[4, 4], color="#6B7686").encode(y="y:Q")
    chart((line + pts + rule).properties(height=260))
    st.caption(t("tl_caption"))

# ==========================================================================
# TAB 3 — EVIDENCE
# ==========================================================================
with tab_ev:
    st.markdown(f'<div class="section-label">{t("ev_alerts_title")}</div>', unsafe_allow_html=True)
    ec = st.columns(4)
    kpi(ec[0], t("m_flagged"), f"{astats['flagged']:,}")
    kpi(ec[1], t("m_high"), f"{astats['high']:,}")
    kpi(ec[2], t("ev_k_admin"), f"{astats['admin_share_pct']:.0f}%")
    kpi(ec[3], t("ev_k_red"), f"-{astats['reduction_pct']:.0f}%")
    st.write("")
    st.write(t("ev_alerts_text", flagged=f"{astats['flagged']:,}", admin_users=astats["admin_users"],
               admin_share=f"{astats['admin_share_pct']:.1f}", admin_routine=f"{astats['admin_routine_pct']:.0f}",
               high=f"{astats['high']:,}", red=f"{astats['reduction_pct']:.0f}"))

    st.markdown(f'<div class="section-label">{t("ev_inj_title")}</div>', unsafe_allow_html=True)
    st.write(t("ev_inj_text", nr=f"{evaluation['n_regular']:,}", na=f"{evaluation['n_admin']:,}"))

    def eval_frame(group: str) -> pd.DataFrame:
        rows = []
        for scen in SCEN_ORDER:
            for lvl in LVL_ORDER:
                v = evaluation[group][scen][lvl]
                rows.append({"scenario": t("ev_scen_" + scen), "level": t("ev_lvl_" + lvl),
                             t("ev_metric_model"): v["recall_model"], t("ev_metric_system"): v["recall_system"]})
        return pd.DataFrame(rows)

    for group, title in [("regular_users", t("ev_regular")), ("admin_users", t("ev_admin"))]:
        st.markdown(f"**{title}**")
        long = eval_frame(group).melt(["scenario", "level"], var_name="approach", value_name="recall")
        lvl_sort = [t("ev_lvl_" + l) for l in LVL_ORDER]
        ch = alt.Chart(long).mark_bar().encode(
            x=alt.X("level:N", sort=lvl_sort, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="approach:N",
            y=alt.Y("recall:Q", title=t("ev_recall"), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("approach:N", title=None,
                            scale=alt.Scale(domain=[t("ev_metric_model"), t("ev_metric_system")], range=["#6B7686", "#22D3EE"])),
            column=alt.Column("scenario:N", title=None, sort=[t("ev_scen_" + s) for s in SCEN_ORDER]),
            tooltip=["scenario", "level", "approach", "recall"],
        ).properties(width=170, height=170)
        st.altair_chart(ch)

    rg, ad = evaluation["regular_users"], evaluation["admin_users"]
    st.markdown(f'<div class="section-label">{t("ev_find_title")}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="explain-box">
<div>› {t("ev_find1", a=rg["after_hours"]["strong"]["recall_system"], b=rg["usb_copy"]["strong"]["recall_system"],
         c=rg["after_hours"]["subtle"]["recall_system"], d=rg["usb_copy"]["subtle"]["recall_system"])}</div>
<div>› {t("ev_find2", w=rg["web_exfil"]["strong"]["recall_system"])}</div>
<div>› {t("ev_find3", m=ad["after_hours"]["strong"]["recall_model"], s=ad["after_hours"]["strong"]["recall_system"])}</div>
</div>""", unsafe_allow_html=True)
    st.caption(t("ev_caveat"))

# ==========================================================================
# TAB 4 — REALITY CHECK
# ==========================================================================
with tab_rc:
    st.markdown(f"### {t('rc_title')}")
    st.write(t("rc_intro"))
    rc1, rc2 = st.columns(2, gap="large")
    for col, key in [(rc1, "aramco"), (rc2, "binance")]:
        with col:
            st.markdown(f"""<div class="card"><div style="font-weight:700;font-size:16px;margin-bottom:10px">{t(f"rc_{key}_t")}</div>
<div class="metric-label">{t("rc_happened")}</div><div style="margin-bottom:10px">{t(f"rc_{key}_h")}</div>
<div class="metric-label">{t("rc_sees")}</div><div style="margin-bottom:10px">{t(f"rc_{key}_s")}</div>
<div class="metric-label">{t("rc_blind")}</div><div>{t(f"rc_{key}_b")}</div></div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="explain-box"><div>{t("rc_takeaway")}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{t("rc_arch_title")}</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#10151F", color="#22D3EE", fontcolor="#E4E9F0", fontname="Helvetica"];
  edge [color="#6B7686", fontcolor="#8A94A6", fontname="Helvetica"];
  src [label="Log sources\\nWindows events · USB/EDR\\nproxy · firewall"];
  siem [label="SIEM / syslog"];
  job [label="Scoring job\\n(scheduled)"];
  db [label="Results DB"];
  dash [label="InsiderGuard\\n(read-only dashboard)"];
  rp [label="Reverse proxy\\nTLS + SSO/AD"];
  soc [label="SOC analysts\\n(SOC subnet only)"];
  src -> siem -> job -> db -> dash -> rp -> soc;
  soc -> job [label="feedback", style=dashed];
}
""")
    st.markdown("".join(f"<div>› {t(f'rc_arch{i}')}</div>" for i in range(1, 6)), unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">{t("rc_sources")}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="ext">
<div><a href="https://www.bankinfosecurity.com/saudi-aramco-says-supplier-leaked-company-data-a-17130" target="_blank">Saudi Aramco says supplier leaked company data (BankInfoSecurity)</a></div>
<div><a href="https://cshub.com/executive-decisions/articles/iotw-contractor-allegedly-responsible-for-aramco-50-million-ransom" target="_blank">Contractor allegedly responsible for Aramco $50 million ransom (CSHub)</a></div>
<div><a href="https://techcrunch.com/2026/11/01/binance-resumes-withdrawals-following-6-3-billion-breach/" target="_blank">Binance resumes withdrawals following $6.3 billion breach (TechCrunch)</a></div>
<div><a href="https://www.sei.cmu.edu/library/insider-threat-test-dataset/" target="_blank">CERT Insider Threat Test Dataset (CMU SEI)</a></div>
</div>""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">{t("footer")}</div>', unsafe_allow_html=True)
