# streamlit_app.py
# App demo: Chấm điểm phê duyệt, gợi ý hạn mức, gợi ý thẻ, DSS (rule-based)

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------- Helpers ---------------------
@st.cache_resource
def load_models():
    clf = joblib.load("models/approval_model.joblib")
    reg = joblib.load("models/limit_model.joblib")
    return clf, reg

@st.cache_data
def load_data():
    df = pd.read_csv("data/synthetic_customers.csv")
    with open("data/card_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
    return df, catalog

def score_cards(row: pd.Series, catalog):
    # Ước lượng utility theo cashback - annual fee (content-based)
    scores = []
    spend_map = {
        "groceries": row["avg_monthly_spend"] * row["spend_groceries"],
        "travel": row["avg_monthly_spend"] * row["spend_travel"],
        "online": row["avg_monthly_spend"] * row["spend_online"],
        "utilities": row["avg_monthly_spend"] * row["spend_utilities"],
        "entertainment": row["avg_monthly_spend"] * row["spend_entertainment"],
    }
    income = row["income"]
    for c in catalog:
        eligibility_penalty = 0.0 if income >= c["min_income"] else -1.0
        est_cashback = 0.0
        for k, v in c["cashback"].items():
            est_cashback += spend_map[k] * v
        net_year = est_cashback * 12 - c["annual_fee"]
        score = net_year / 1_000_000 + eligibility_penalty  # đơn vị: triệu VND
        scores.append({**c, "utility_score": score, "est_cashback_year": est_cashback * 12})
    return sorted(scores, key=lambda x: x["utility_score"], reverse=True)

def dss_policy(prob_approval, suggested_limit, row):
    """
    Lớp chính sách demo (KHÔNG phải chính sách TPBank):
    - Ngưỡng PD để 'đủ điều kiện' (0.55)
    - Bác nếu trễ hạn nhiều hoặc điểm tín dụng thấp
    - Cảnh báo utilization cao
    - Clamp hạn mức theo thu nhập
    """
    reasons = []
    approved = prob_approval >= 0.55
    max_limit = min(row["income"] * 5, 300_000_000)

    if row["late_payments_12m"] >= 3 or row["credit_score"] < 560:
        approved = False
        reasons.append("Nợ quá hạn nhiều/điểm tín dụng thấp")
    if row["utilization_rate"] > 0.85:
        reasons.append("Tỷ lệ sử dụng thẻ hiện tại cao")
    if row["income"] < 7_000_000:
        reasons.append("Thu nhập dưới ngưỡng tối thiểu")

    limit = int(np.clip(suggested_limit, 5_000_000, max_limit))
    return approved, limit, reasons

# --------------------- App ---------------------
st.set_page_config(page_title="TPBank Credit – ML + RS + DSS Demo", layout="wide")
st.title("Gợi ý mở thẻ/nâng hạn mức áp dụng ML")
st.caption("⚠️ Demo học thuật, KHÔNG phản ánh chính sách chính thức của TPBank.")

df, catalog = load_data()
clf, reg = load_models()

left, right = st.columns([0.42, 0.58])

with left:
    st.subheader("1) Chọn khách hàng")
    idx = st.number_input("Index khách hàng (0..n-1)", min_value=0, max_value=len(df)-1, value=0, step=1)
    row = df.iloc[int(idx)].copy()
    st.write("**Customer ID:**", row["customer_id"])

    editable = st.checkbox("Bật What-If (chỉnh tham số)")

    fields = ["income","credit_score","avg_monthly_spend","utilization_rate",
              "late_payments_12m","current_limit","employment_length"]
    if editable:
        row["income"] = st.number_input("income", min_value=0, value=int(row["income"]), step=500_000)
        row["credit_score"] = st.number_input("credit_score", min_value=300, max_value=850, value=int(row["credit_score"]), step=5)
        row["avg_monthly_spend"] = st.number_input("avg_monthly_spend", min_value=0, value=int(row["avg_monthly_spend"]), step=200_000)
        row["utilization_rate"] = st.number_input("utilization_rate", min_value=0.0, max_value=1.0, value=float(row["utilization_rate"]), step=0.01)
        row["late_payments_12m"] = st.number_input("late_payments_12m", min_value=0, max_value=12, value=int(row["late_payments_12m"]), step=1)
        row["current_limit"] = st.number_input("current_limit", min_value=0, value=int(row["current_limit"]), step=1_000_000)
        row["employment_length"] = st.number_input("employment_length", min_value=0, max_value=40, value=int(row["employment_length"]), step=1)

with right:
    st.subheader("2) Chấm điểm & gợi ý")
    features = ["age","income","employment_length","credit_score","existing_cc_count","avg_monthly_spend",
                "spend_groceries","spend_travel","spend_online","spend_utilities","spend_entertainment",
                "tenure_months","late_payments_12m","current_limit","utilization_rate",
                "has_mortgage","has_auto_loan","employment_type","region"]

    X = pd.DataFrame([row[features]], columns=features)
    prob_approval = float(clf.predict_proba(X)[0, 1])
    st.metric("Xác suất phê duyệt (PD*)", f"{prob_approval:.2f}")
    st.caption("*PD ở đây là xác suất được phê duyệt (probability of approval).")

    suggested_limit = float(reg.predict(X)[0])
    st.metric("Hạn mức đề xuất (model)", f"{int(suggested_limit):,} đ")

    approved, limit, reasons = dss_policy(prob_approval, suggested_limit, row)
    st.markdown("**Kết luận DSS:** " + ("✅ Đủ điều kiện" if approved else "❌ Chưa đủ điều kiện"))
    st.markdown(f"**Hạn mức sau kiểm soát DSS:** {limit:,} đ")
    if reasons:
        with st.expander("Lý do/giải trình theo quy tắc"):
            for r in reasons:
                st.write("- " + r)

    st.subheader("3) Gợi ý thẻ phù hợp (Recommender)")
    recs = score_cards(row, catalog)
    topk = recs[:3]
    st.dataframe(pd.DataFrame([{
        "Mã thẻ": r["card_code"],
        "Tên": r["name"],
        "Phí thường niên": r["annual_fee"],
        "Ước tính cashback/năm": int(r["est_cashback_year"]),
        "Điểm hữu ích (triệu)": round(r["utility_score"], 2),
        "Yêu cầu thu nhập": r["min_income"],
        "Phân khúc": r["target"]
    } for r in topk]), use_container_width=True)

    st.subheader("4) Xuất báo cáo quyết định")
    report = {
        "customer_id": row["customer_id"],
        "prob_approval": prob_approval,
        "suggested_limit_model": int(suggested_limit),
        "dss_approved": approved,
        "dss_limit": limit,
        "reasons": reasons,
        "top_cards": [{"card_code": r["card_code"], "name": r["name"], "utility_score": r["utility_score"]} for r in topk]
    }
    st.download_button(
        "Tải report.json",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=f"{row['customer_id']}_decision.json",
        mime="application/json"
    )

st.divider()
st.caption("© Demo học thuật. ML: RandomForest; RS: content-based theo hành vi chi tiêu; DSS: rule-based.")
