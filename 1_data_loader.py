"""
الخطوة 1: تحميل واستكشاف البيانات
=====================================
حط ملفات الـ CSV (logon.csv, device.csv, http.csv, file.csv, email.csv)
داخل مجلد data/ اللي جنب هذا الملف، ثم شغّل:

    python 1_data_loader.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

FILES = ["logon.csv", "device.csv", "http.csv", "file.csv", "email.csv"]


def load_all():
    """يحمّل كل ملف موجود ويطلع لك ملخص سريع عنه."""
    dataframes = {}
    for fname in FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"⚠️  {fname} غير موجود في مجلد data/ — تجاهلته")
            continue

        # بعض الملفات (زي http.csv) ما فيها سطر عناوين، فنحطه يدويًا
        expected_cols = ["id", "date", "user", "pc", "activity_or_url"]
        first_line = fpath.open(encoding="utf-8", errors="ignore").readline()
        has_header = "date" in first_line.lower() or "user" in first_line.lower()

        if has_header:
            df = pd.read_csv(fpath)
        else:
            df = pd.read_csv(fpath, header=None, names=expected_cols)

        dataframes[fname.replace(".csv", "")] = df

        print(f"\n{'='*50}")
        print(f"📄 {fname}")
        print(f"{'='*50}")
        print(f"عدد الصفوف: {len(df):,}")
        print(f"الأعمدة: {list(df.columns)}")
        print(f"عدد المستخدمين الفريدين: {df['user'].nunique() if 'user' in df.columns else 'N/A'}")
        print(df.head(3))

    return dataframes


if __name__ == "__main__":
    data = load_all()
    print(f"\n✅ تم تحميل {len(data)} ملف بنجاح.")
