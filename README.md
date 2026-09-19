# boAt Order Risk Scoring — Predictive Analytics Dashboard

Interactive Streamlit dashboard performing exploratory analysis and live return/cancellation risk scoring on boAt's e-commerce sales dataset (~975K orders).

## 🔗 Live app

URL: https://your-app-name.streamlit.app/

### Credentials

- Username: `username`
- Password: `boat risk scoring`

## 📊 What's inside

- **Explore & Filter tab** — headline KPIs, risk score distribution, return/cancellation rate by category, filterable order table
- **Score a New Order tab** — gradient boosting classifier with an interactive scoring form for live risk prediction on new orders

## 🛠️ Tech stack

- Python · pandas · scikit-learn · joblib
- Plotly
- Streamlit

## 📁 Repository structure

```
boat-risk-app/
├── app.py                    # Streamlit app (login + dashboard + scoring)
├── train_model.ipynb         # Model training notebook
├── requirements.txt          # Python dependencies
├── runtime.txt                # Pinned Python version
├── risk_model.pkl             # Trained scikit-learn pipeline
├── categorical_options.pkl    # Dropdown options for the scoring form
├── data/
│   └── boat_scored_orders.csv.gz
└── README.md
```


## 🧠 Model summary

- **Type:** Gradient Boosting Classifier (scikit-learn), trained as a lightweight equivalent of a PySpark MLlib GBT model built in Databricks
- **Target:** binary — did the order get returned or cancelled?
- **Features:** CustomerTier, CustomerSegment, CityTier, ChannelType, PaymentMethod, ProductCategory, OrderHour, Quantity, GrossMRPValue, DiscountPct, IsFestivalPeriod, IsWeekend, HeroFlag
- **Preprocessing:** One-hot encoding + passthrough numeric features, in a scikit-learn Pipeline
- **Test-set ROC-AUC:** ROC-AUC: 0.5806

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```