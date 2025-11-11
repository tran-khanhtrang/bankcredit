# build_demo.py
# Tạo dataset giả lập, train model phê duyệt & hạn mức, lưu vào thư mục ./data và ./models

import os, json, random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

# -------------------- Chuẩn bị thư mục --------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------- Sinh dữ liệu giả lập --------------------
np.random.seed(42)
random.seed(42)
N = 1500

regions = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]
employment_types = ["Công chức", "Nhân viên văn phòng", "Kỹ sư CNTT", "Tự doanh", "Freelancer"]

def gen_customers(n=1500):
    rows = []
    for i in range(n):
        age = np.random.randint(21, 65)
        income = np.random.randint(6, 120) * 1_000_000      # VND/tháng (ước lượng)
        emp_len = np.random.randint(0, 35)
        emp_type = random.choice(employment_types)
        credit_score = np.clip(int(np.random.normal(680, 70)), 300, 850)
        existing_cc = np.random.poisson(1.2)
        spend_total = np.random.randint(2_000_000, 45_000_000)
        # tỉ trọng chi tiêu theo nhóm, tổng ~ 1
        gro, travel, online, utilities, ent = np.random.dirichlet([1.5, 1.0, 1.2, 0.8, 0.9])
        late_12m = np.random.binomial(4, 0.12)
        tenure = np.random.randint(1, 140)                  # tháng
        curr_limit = np.random.randint(10_000_000, 200_000_000)
        util = float(np.clip(np.random.beta(2, 6), 0, 0.98))
        has_mortgage = np.random.binomial(1, 0.25)
        has_auto = np.random.binomial(1, 0.20)
        region = random.choice(regions)

        # proxy debt-to-income (đơn giản hoá)
        dti_proxy = (spend_total + (has_mortgage*5_000_000) + (has_auto*3_000_000)) / max(income, 1)
        base_pd = 0.12 + 0.25*(util>0.6) + 0.2*(late_12m>=2) + 0.2*(credit_score<600) + 0.1*(income<8_000_000) + 0.1*(dti_proxy>0.6)
        base_pd = float(np.clip(base_pd, 0.01, 0.95))
        approve = np.random.binomial(1, 1 - base_pd)

        # “ground truth” hạn mức (ẩn) để huấn luyện hồi quy
        limit_truth = int(np.clip(income*3.5*(credit_score/750)*(0.9 if late_12m else 1.0), 5_000_000, 300_000_000))

        rows.append(dict(
            customer_id=f"C{i:05d}",
            age=age, income=income, employment_length=emp_len, employment_type=emp_type,
            credit_score=credit_score, existing_cc_count=existing_cc, avg_monthly_spend=spend_total,
            spend_groceries=gro, spend_travel=travel, spend_online=online,
            spend_utilities=utilities, spend_entertainment=ent,
            tenure_months=tenure, late_payments_12m=late_12m, current_limit=curr_limit,
            utilization_rate=util, has_mortgage=has_mortgage, has_auto_loan=has_auto,
            region=region, approved=approve, true_limit=limit_truth
        ))
    return pd.DataFrame(rows)

df = gen_customers(N)
df.to_csv("data/synthetic_customers.csv", index=False)
print(f"[OK] generated data/synthetic_customers.csv with {len(df)} rows")

# -------------------- Danh mục thẻ (demo) --------------------
card_catalog = [
    {"card_code": "TPB_GO", "name": "TPBank GO", "annual_fee": 400_000,
     "cashback": {"groceries": 0.02, "travel": 0.00, "online": 0.01, "utilities": 0.00, "entertainment": 0.01},
     "min_income": 8_000_000, "target": "daily"},
    {"card_code": "TPB_STEPUP", "name": "TPBank Step Up", "annual_fee": 990_000,
     "cashback": {"groceries": 0.01, "travel": 0.02, "online": 0.02, "utilities": 0.00, "entertainment": 0.02},
     "min_income": 12_000_000, "target": "young_pro"},
    {"card_code": "TPB_PLATINUM", "name": "TPBank Platinum", "annual_fee": 1_200_000,
     "cashback": {"groceries": 0.01, "travel": 0.03, "online": 0.02, "utilities": 0.00, "entertainment": 0.02},
     "min_income": 18_000_000, "target": "affluent"},
    {"card_code": "TPB_FREE", "name": "TPBank Free", "annual_fee": 0,
     "cashback": {"groceries": 0.005, "travel": 0.0, "online": 0.005, "utilities": 0.0, "entertainment": 0.005},
     "min_income": 6_000_000, "target": "starter"},
]
with open("data/card_catalog.json", "w", encoding="utf-8") as f:
    json.dump(card_catalog, f, ensure_ascii=False, indent=2)
print("[OK] created data/card_catalog.json")

# -------------------- Huấn luyện model --------------------
num_features = [
    "age","income","employment_length","credit_score","existing_cc_count","avg_monthly_spend",
    "spend_groceries","spend_travel","spend_online","spend_utilities","spend_entertainment",
    "tenure_months","late_payments_12m","current_limit","utilization_rate",
    "has_mortgage","has_auto_loan"
]
cat_features = ["employment_type","region"]

X = df[num_features + cat_features]
y_cls = df["approved"]
y_reg = df["true_limit"]

preprocess = ColumnTransformer([
    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_features),
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_features)
], remainder="drop")

clf = Pipeline([("pre", preprocess),
                ("rf", RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced"))])

reg = Pipeline([("pre", preprocess),
                ("rf", RandomForestRegressor(n_estimators=280, random_state=42))])

X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
Xr_train, Xr_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

clf.fit(X_train, y_cls_train)
reg.fit(Xr_train, y_reg_train)

proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_cls_test, proba)
acc = accuracy_score(y_cls_test, (proba >= 0.5).astype(int))
pred_limit = reg.predict(Xr_test)
mae = mean_absolute_error(y_reg_test, pred_limit)
r2 = r2_score(y_reg_test, pred_limit)

metrics = {
    "classification_auc": float(auc),
    "classification_accuracy": float(acc),
    "regression_mae": float(mae),
    "regression_r2": float(r2)
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(clf, "models/approval_model.joblib")
joblib.dump(reg, "models/limit_model.joblib")

print("[OK] trained models saved to ./models")
print("[METRICS]", json.dumps(metrics, indent=2))
