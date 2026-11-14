"""
الخطوة 3: بناء نموذج كشف الشذوذ + درجة الخطورة
==================================================
يقرأ daily_features.csv (الناتج من الخطوة 2) ويبني نموذج
Isolation Forest يكتشف السلوك غير الطبيعي، ثم يحوّل النتيجة
إلى "درجة خطورة" من 0 إلى 100 لكل مستخدم-يوم.

شغّله بعد 2_feature_engineering.py:

    python 3_train_model.py
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

FEATURES_PATH = Path(__file__).parent / "daily_features.csv"
OUT_PATH = Path(__file__).parent / "risk_scores.csv"

FEATURE_COLS = [
    "logon_count", "off_hours_logons", "usb_events",
    "http_events", "file_events", "email_events",
]


def train_and_score():
    df = pd.read_csv(FEATURES_PATH)

    # يستخدم بس الأعمدة الموجودة فعليًا (بعض الملفات قد ما تكون متوفرة)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # نتوقع ~2% من الأيام شاذة
        random_state=42,
    )
    model.fit(X)

    # decision_function: كل ما قلّت القيمة، كل ما كان أشذ
    raw_scores = model.decision_function(X)

    # نحوّلها لمقياس 0-100 (100 = أخطر)
    df["risk_score"] = (
        (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min()) * 100
    ).round(1)

    df["is_anomaly"] = model.predict(X) == -1

    df_sorted = df.sort_values("risk_score", ascending=False)
    df_sorted.to_csv(OUT_PATH, index=False)

    print(f"✅ تم حساب درجات الخطورة: {OUT_PATH}")
    print(f"\n🔴 أعلى 10 حالات خطورة:")
    print(df_sorted[["user", "date_only", "risk_score"] + cols].head(10).to_string(index=False))

    return df_sorted


if __name__ == "__main__":
    train_and_score()
