"""
الخطوة 2: هندسة الميزات (Feature Engineering)
=================================================
يحوّل السجلات الخام (كل صف = حدث واحد) إلى جدول واحد
شكله: كل صف = (مستخدم + يوم) مع أرقام تلخّص نشاطه ذاك اليوم.

شغّله بعد ما يشتغل عندك 1_data_loader.py بدون أخطاء:

    python 2_feature_engineering.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "daily_features.csv"


def smart_read_csv(fpath):
    """يقرأ CSV ويتأكد من وجود عمود 'date'، حتى لو الملف بدون سطر عناوين."""
    first_line = fpath.open(encoding="utf-8", errors="ignore").readline()
    has_header = "date" in first_line.lower() or "user" in first_line.lower()
    if has_header:
        return pd.read_csv(fpath)
    return pd.read_csv(
        fpath, header=None,
        names=["id", "date", "user", "pc", "activity_or_url"],
    )


def add_date_col(df):
    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    return df


def count_per_user_day(df, colname):
    """يرجع Series: (user, date) -> عدد الأحداث"""
    df = add_date_col(df)
    return df.groupby(["user", "date_only"]).size().rename(colname)


def logon_off_hours(logon_df):
    """يحسب عدد مرات الدخول خارج ساعات الدوام الطبيعية (قبل 6ص أو بعد 8م)."""
    df = add_date_col(logon_df)
    df["hour"] = pd.to_datetime(df["date"]).dt.hour
    off_hours = df[(df["hour"] < 6) | (df["hour"] > 20)]
    return off_hours.groupby(["user", "date_only"]).size().rename("off_hours_logons")


def build_features():
    features = []

    logon_path = DATA_DIR / "logon.csv"
    if logon_path.exists():
        logon = smart_read_csv(logon_path)
        features.append(count_per_user_day(logon, "logon_count"))
        features.append(logon_off_hours(logon))

    device_path = DATA_DIR / "device.csv"
    if device_path.exists():
        device = smart_read_csv(device_path)
        features.append(count_per_user_day(device, "usb_events"))

    http_path = DATA_DIR / "http.csv"
    if http_path.exists():
        http = smart_read_csv(http_path)
        features.append(count_per_user_day(http, "http_events"))

    file_path = DATA_DIR / "file.csv"
    if file_path.exists():
        file_df = smart_read_csv(file_path)
        features.append(count_per_user_day(file_df, "file_events"))

    email_path = DATA_DIR / "email.csv"
    if email_path.exists():
        email = smart_read_csv(email_path)
        features.append(count_per_user_day(email, "email_events"))

    if not features:
        raise SystemExit("❌ ما لقيت أي ملف بيانات في مجلد data/. حط الملفات أول.")

    result = pd.concat(features, axis=1).fillna(0).reset_index()
    result.to_csv(OUT_PATH, index=False)
    print(f"✅ تم بناء الميزات اليومية: {OUT_PATH}")
    print(f"عدد الصفوف (مستخدم-يوم): {len(result):,}")
    print(result.head(10))
    return result


if __name__ == "__main__":
    build_features()
