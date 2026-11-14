"""
Offline pipeline: daily features -> model + scored table + evidence files.

    python scripts/build_artifacts.py \
        --features ../hackathon/files/daily_features.csv \
        --ldap ../hackathon/files/data/LDAP

Outputs (in ./data):
    scores.parquet   one row per user-day: features, risk, flag, own baseline, priority
    model.joblib     the trained Isolation Forest (+ score normalisation)
    users.csv        user -> LDAP role
    eval.json        red-team injection test + alert-volume numbers
    cases.json       cases for the "analyst" experience
    personas.json    employees for the "insider" experience
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import engine as E  # noqa: E402

RNG = np.random.default_rng(7)

# Red-team scenarios: additive deltas on a normal day, per intensity.
# order of deltas: logon_count, off_hours_logons, usb_events, http_events
SCENARIOS = {
    "after_hours": {
        "subtle":   [2, 2, 0, 0],
        "moderate": [4, 4, 0, 0],
        "strong":   [8, 8, 0, 0],
    },
    "usb_copy": {
        "subtle":   [0, 0, 2, 0],
        "moderate": [0, 1, 4, 10],
        "strong":   [2, 2, 6, 30],
    },
    "web_exfil": {
        "subtle":   [0, 0, 0, 30],
        "moderate": [0, 0, 0, 100],
        "strong":   [0, 0, 0, 250],
    },
}


def load_roles(ldap_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(ldap_dir / "*.csv"))):
        d = pd.read_csv(f)
        d["month"] = Path(f).stem
        frames.append(d)
    ld = pd.concat(frames)
    ld = ld.sort_values("month").groupby("user_id").tail(1)
    out = ld[["user_id", "Role"]].rename(columns={"user_id": "uid", "Role": "role"})
    out["user"] = "DTAA/" + out["uid"]
    return out[["user", "role"]].reset_index(drop=True)


def priority_col(flagged: np.ndarray, unusual_any: np.ndarray, has_base: np.ndarray) -> np.ndarray:
    out = np.full(len(flagged), "", dtype=object)
    out[flagged & ~has_base] = "HIGH"
    out[flagged & has_base & unusual_any] = "HIGH"
    out[flagged & has_base & ~unusual_any] = "ROUTINE"
    return out


def inject(base_x: np.ndarray, delta) -> np.ndarray:
    return base_x + np.asarray(delta, float)


def run_injection(bundle, rows: pd.DataFrame, label: str) -> dict:
    X = rows[E.FEATURES].to_numpy(float)
    med, p95 = E.baseline_arrays(rows)
    res = {}
    for scen, levels in SCENARIOS.items():
        res[scen] = {}
        for lvl, delta in levels.items():
            Xi = inject(X, delta)
            risk, flagged = E.score_matrix(bundle, Xi)
            unusual = E.unusual_flags(Xi, p95).any(axis=1)
            high = flagged & unusual
            res[scen][lvl] = {
                "recall_model": round(float(flagged.mean()) * 100, 1),
                "recall_system": round(float(high.mean()) * 100, 1),
                "mean_risk": round(float(risk.mean()), 1),
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--ldap", required=True)
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading features ...")
    df = pd.read_csv(a.features)
    df[E.FEATURES] = df[E.FEATURES].fillna(0)
    print(f"  {len(df):,} user-days, {df.user.nunique()} users")

    print("training model ...")
    bundle = E.train_bundle(df)
    raw = bundle["model"].decision_function(df[E.FEATURES])
    df["risk_score"] = E.risk_from_raw(raw, bundle).round(1)
    df["is_anomaly"] = bundle["model"].predict(df[E.FEATURES]) == -1
    joblib.dump(bundle, out / "model.joblib", compress=3)

    print("building per-user baselines ...")
    df = E.build_baselines(df)
    med, p95 = E.baseline_arrays(df)
    has_base = ~np.isnan(p95).any(axis=1)
    unusual = np.zeros(len(df), dtype=bool)
    unusual[has_base] = E.unusual_flags(df.loc[has_base, E.FEATURES].to_numpy(float), p95[has_base]).any(axis=1)
    df["unusual_for_user"] = unusual
    df["priority"] = priority_col(df["is_anomaly"].to_numpy(), unusual, has_base)

    roles = load_roles(Path(a.ldap))
    roles.to_csv(out / "users.csv", index=False)
    df = df.merge(roles, on="user", how="left")
    df["role"] = df["role"].fillna("Unknown")

    # ---- alert-volume numbers (real data) ----
    fl = df[df.is_anomaly]
    high = fl[fl.priority == "HIGH"]
    routine = fl[fl.priority == "ROUTINE"]
    admins = fl[fl.role == "IT Admin"]
    alert_stats = {
        "records": int(len(df)),
        "users": int(df.user.nunique()),
        "flagged": int(len(fl)),
        "flagged_users": int(fl.user.nunique()),
        "high": int(len(high)),
        "high_users": int(high.user.nunique()),
        "routine": int(len(routine)),
        "admin_flagged": int(len(admins)),
        "admin_share_pct": round(len(admins) / len(fl) * 100, 1),
        "admin_users": int(admins.user.nunique()),
        "admin_routine_pct": round((admins.priority == "ROUTINE").mean() * 100, 1),
        "reduction_pct": round((1 - len(high) / len(fl)) * 100, 1),
        "peak_risk": float(df.risk_score.max()),
    }
    print("  alert stats:", alert_stats)

    # ---- red-team injection test ----
    print("running red-team injection test ...")
    ok = df[(~df.is_anomaly) & has_base].copy()
    non_admin = ok[ok.role != "IT Admin"]
    admin = ok[ok.role == "IT Admin"]
    s_non = non_admin.sample(3000, random_state=11)
    s_adm = admin.sample(min(1500, len(admin)), random_state=11)
    evaluation = {
        "n_regular": int(len(s_non)),
        "n_admin": int(len(s_adm)),
        "regular_users": run_injection(bundle, s_non, "regular"),
        "admin_users": run_injection(bundle, s_adm, "admin"),
        "scenarios": SCENARIOS,
        "alert_stats": alert_stats,
    }

    # ---- personas (whole-history profile) ----
    print("building personas ...")
    prof = df.groupby("user").agg(
        **{f"med_{f}": (f, "median") for f in E.FEATURES},
        **{f"p95_{f}": (f, lambda s: float(np.quantile(s, 0.95))) for f in E.FEATURES},
        mean_http=("http_events", "mean"),
        days=("risk_score", "size"),
        flagged=("is_anomaly", "sum"),
    ).reset_index().merge(roles, on="user", how="left")
    reg = prof[(prof.role != "IT Admin") & (prof.days >= 300)].copy()
    calm = reg[reg.flagged <= 25]

    def pick(sel: pd.DataFrame, key: str, ascending=False):
        return sel.sort_values(key, ascending=ascending).iloc[0]

    usb_users = calm[calm.p95_usb_events >= 2]
    web_heavy = calm.sort_values("mean_http", ascending=False).head(20)
    night = calm[calm.p95_off_hours_logons >= 2]
    plain = calm[(calm.p95_usb_events == 0) & (calm.p95_off_hours_logons == 0)
                 & (calm.mean_http.between(4, 8))]
    adm = prof[prof.role == "IT Admin"].sort_values("flagged", ascending=False)

    def persona(row, kind):
        return {
            "kind": kind,
            "user": row["user"],
            "role": row["role"],
            "median": [float(row[f"med_{f}"]) for f in E.FEATURES],
            "p95": [float(row[f"p95_{f}"]) for f in E.FEATURES],
        }

    personas = [
        persona(plain.sort_values("days", ascending=False).iloc[0], "office"),
        persona(pick(usb_users, "p95_usb_events"), "usb"),
        persona(pick(web_heavy, "mean_http"), "web"),
        persona(pick(night, "p95_off_hours_logons"), "night"),
        persona(adm.iloc[0], "admin"),
    ]

    # ---- cases for the analyst experience ----
    print("building analyst cases ...")
    cases = []

    def case_from_row(r, truth, scenario, note_key):
        xs = r[E.FEATURES].to_numpy(float)
        med_r = np.array([r[f"b_med_{f}"] for f in E.FEATURES], float)
        p95_r = np.array([r[f"b_p95_{f}"] for f in E.FEATURES], float)
        risk, flagged = E.score_matrix(bundle, xs)
        unusual_any = bool(E.unusual_flags(xs, p95_r).any())
        return {
            "user": r["user"], "role": r["role"],
            "date": str(r["date_only"]) if "date_only" in r else "",
            "x": [float(v) for v in xs],
            "median": [float(v) for v in med_r],
            "p95": [float(v) for v in p95_r],
            "risk": round(float(risk[0]), 1),
            "flagged": bool(flagged[0]),
            "priority": E.priority(bool(flagged[0]), unusual_any),
            "truth": truth, "scenario": scenario, "note": note_key,
        }

    # 1) highest-risk day that is routine for an IT admin (their normal job)
    top = admins[admins.priority == "ROUTINE"].sort_values("risk_score", ascending=False).iloc[0]
    cases.append(case_from_row(top, "routine", "", "admin_top"))
    # 2) a flagged day that is routine for a non-admin user
    r2 = fl[(fl.priority == "ROUTINE") & (fl.role != "IT Admin")].sort_values("risk_score", ascending=False)
    if len(r2):
        cases.append(case_from_row(r2.iloc[0], "routine", "", "routine_user"))
    else:
        cases.append(case_from_row(admins[admins.priority == "ROUTINE"].iloc[5], "routine", "", "admin_routine"))
    # 3 + 4) red-team scenarios injected into ordinary people's ordinary days
    pool = non_admin[(non_admin.risk_score < 40)].sample(400, random_state=5)
    for scen, lvl, note in [("usb_copy", "strong", "attack_usb"), ("web_exfil", "strong", "attack_web")]:
        for _, r in pool.iterrows():
            if r["user"] in {c["user"] for c in cases}:
                continue
            rr = r.copy()
            xs = r[E.FEATURES].to_numpy(float) + np.asarray(SCENARIOS[scen][lvl], float)
            for i, f in enumerate(E.FEATURES):
                rr[f] = xs[i]
            c = case_from_row(rr, "attack", f"{scen}:{lvl}", note)
            if c["priority"] == "HIGH":
                cases.append(c)
                break

    # ---- save ----
    keep = ["user", "date_only"] + E.FEATURES + ["risk_score", "is_anomaly", "priority", "role"] \
        + [f"b_med_{f}" for f in E.FEATURES] + [f"b_p95_{f}" for f in E.FEATURES]
    save = df[keep].copy()
    for f in E.FEATURES:
        save[f] = save[f].astype("float32")
    save["risk_score"] = save["risk_score"].astype("float32")
    save["user"] = save["user"].astype("category")
    save["role"] = save["role"].astype("category")
    save["priority"] = save["priority"].astype("category")
    save["date_only"] = save["date_only"].astype(str)
    save.to_parquet(out / "scores.parquet", index=False, compression="zstd")

    (out / "eval.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out / "personas.json").write_text(json.dumps(personas, indent=2), encoding="utf-8")
    (out / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print("done.")
    for p in sorted(out.glob("*")):
        print(f"  {p.name:18s} {p.stat().st_size/1e6:7.2f} MB")


if __name__ == "__main__":
    main()
