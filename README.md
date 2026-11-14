# InsiderGuard — كشف التهديدات الداخلية

موقع تفاعلي يحلل سلوك المستخدمين (دخول، USB، ويب) ويعطي كل مستخدم-يوم درجة خطر 0–100، ويشرح السبب،
ويقارن اليوم بتاريخ المستخدم نفسه عشان يقلل الإنذارات الكاذبة.

Interactive insider-threat monitor: Isolation Forest risk scores, plain-language explanations,
and a per-user baseline that separates unusual days from routine ones.

## التشغيل محليًا

```bash
pip install -r requirements.txt
streamlit run app.py
```

## النشر على Streamlit Community Cloud (مجاني)

1. أنشئ مستودع GitHub جديد (Public) باسم `insiderguard`.
2. ارفع كل محتويات هذا المجلد (بما فيها مجلد `data` و`.streamlit`).
3. افتح https://share.streamlit.io ← سجّل بحساب GitHub ← **Create app**.
4. اختر المستودع، الفرع `main`، والملف الرئيسي `app.py` ← **Deploy**.
5. بعد الدقائق الأولى يطلع لك رابط عام. افتحه قبل العرض بدقيقتين عشان يصحى لو كان نايم.

## هيكل المشروع

```
app.py                      الواجهة (تجربة المحكّم، اللوحة، الأدلة، فحص الواقع)
engine.py                   النموذج، التقييم، الأولوية حسب السياق، الشرح
i18n.py                     نصوص الواجهة (English / العربية)
data/                       النموذج والنتائج الجاهزة (خفيفة، تكفي للنشر)
scripts/build_artifacts.py  يعيد بناء ملفات data/ من daily_features.csv
```

## إعادة بناء الملفات (اختياري)

```bash
python scripts/build_artifacts.py --features path/to/daily_features.csv --ldap path/to/LDAP
```

## ملاحظات

- البيانات: CERT Insider Threat Test Dataset r1 (CMU SEI)، وهي اصطناعية وبلا تصنيفات مؤكدة للمهاجمين.
- أرقام تبويب «الأدلة» تأتي من هجمات محاكاة حقناها نحن، فهي تقيس الحساسية وليست دقة في الواقع.
- لو اختلف إصدار scikit-learn عن اللي بُني به `data/model.joblib`، يعيد التطبيق تدريب النموذج تلقائيًا
  بنفس الإعدادات (لحظات في أول تشغيل).
