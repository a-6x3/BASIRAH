# مشروع كشف التهديدات الداخلية (Insider Threat Detection)

## التثبيت (مرة وحدة بس)

```bash
pip install pandas scikit-learn streamlit
```

## طريقة التشغيل بالترتيب

1. **حط ملفات CSV** (logon.csv, device.csv, http.csv, file.csv, email.csv)
   داخل مجلد `data/`

2. **تأكد إن البيانات تنقرأ صح:**
   ```bash
   python 1_data_loader.py
   ```

3. **ابني الميزات اليومية:**
   ```bash
   python 2_feature_engineering.py
   ```
   → ينتج ملف `daily_features.csv`

4. **درّب النموذج واحسب درجات الخطورة:**
   ```bash
   python 3_train_model.py
   ```
   → ينتج ملف `risk_scores.csv` + يطبع لك أعلى 10 حالات خطورة

5. **افتح لوحة التحكم:**
   ```bash
   streamlit run 4_dashboard.py
   ```
   → يفتح متصفح فيه لوحة تفاعلية

## هيكل المشروع

```
insider_threat_project/
├── data/                      ← حط ملفات CSV هنا
├── 1_data_loader.py           ← تحميل واستكشاف
├── 2_feature_engineering.py   ← بناء الميزات
├── 3_train_model.py           ← النموذج + درجة الخطورة
├── 4_dashboard.py             ← لوحة التحكم
└── README.md
```

## ملاحظة

لو ملف من الملفات (مثلاً email.csv) غير موجود، الكود يتجاهله تلقائيًا
ويكمل بالباقي — ما راح يتوقف بخطأ.
