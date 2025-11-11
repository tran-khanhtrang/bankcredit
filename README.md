
# Bank Credit Card – ML + RS + DSS (Demo)

**Lưu ý:** Đây là demo học thuật, KHÔNG đại diện cho chính sách chính thức của TPBank.

## Thành phần
- **ML (phê duyệt):** RandomForestClassifier → xác suất chấp thuận (probability of approval).
- **ML (hạn mức):** RandomForestRegressor → gợi ý hạn mức ban đầu.
- **RS (gợi ý thẻ):** Content-based dựa trên cơ cấu chi tiêu và mức phí/ưu đãi thẻ.
- **DSS (ra quyết định):** Rule-engine đơn giản (ngưỡng PD, tối đa theo thu nhập, lịch sử chậm trả...).

## Cấu trúc
```
bankcredit/
  app/
    streamlit_app.py
  data/
    synthetic_customers.csv
    card_catalog.json
  models/
    approval_model.joblib
    limit_model.joblib
    metrics.json
```
## Chạy nhanh
```bash
pip install -r requirements.txt
cd bankcredit
streamlit run streamlit_app.py

```
## Kết quả model (baseline)
{
  "classification_auc": 0.6175792897804283,
  "classification_accuracy": 0.79,
  "regression_mae": 4422359.695428572,
  "regression_r2": 0.9955704205575855
}
