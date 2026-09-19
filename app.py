import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

import streamlit as st

# ============================================================
# LOGIN GATE
# ============================================================
def check_login():
    def login_form():
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=validate_login)

    def validate_login():
        if (
            st.session_state["username"] == "shitanshu"
            and st.session_state["password"] == "boat_risk_scoring"
        ):
            st.session_state["logged_in"] = True
            del st.session_state["password"]
        else:
            st.session_state["logged_in"] = False
            st.error("Incorrect username or password")

    if st.session_state.get("logged_in", False):
        return True

    st.title("⚡ boAt Order Risk Scoring")
    st.markdown("Please log in to continue.")
    login_form()
    return False

if not check_login():
    st.stop()

# Page config
st.set_page_config(
    page_title="boAt Order Risk Scoring",
    page_icon="⚡",
    layout="wide"
)

# Load model and data once, cached so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    model = joblib.load("risk_model.pkl")
    options = joblib.load("categorical_options.pkl")
    return model, options

@st.cache_data
def load_data():
    return pd.read_csv('data/boAt_scored_orders.csv.gz')

model, categorical_options = load_model()
df = load_data()

# Header
st.title("⚡ boAt Order Risk Scoring")
st.markdown("Interactive tool to explore order data and score new orders for return/cancellation risk.")

tab1, tab2 = st.tabs(["📊 Explore & Filter", "🎯 Score a New Order"])

# ============================================================
# TAB 1: Explore & Filter
# ============================================================
with tab1:
    st.header("Explore Historical Orders")

    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.multiselect(
            "Category", options=sorted(df["ProductCategory"].unique()),
            default=None
        )
    with col2:
        channel_filter = st.multiselect(
            "Channel Type", options=sorted(df["ChannelType"].unique()),
            default=None
        )
    with col3:
        tier_filter = st.multiselect(
            "Customer Tier", options=sorted(df["CustomerTier"].unique()),
            default=None
        )

    # Apply filters
    filtered_df = df.copy()
    if category_filter:
        filtered_df = filtered_df[filtered_df["ProductCategory"].isin(category_filter)]
    if channel_filter:
        filtered_df = filtered_df[filtered_df["ChannelType"].isin(channel_filter)]
    if tier_filter:
        filtered_df = filtered_df[filtered_df["CustomerTier"].isin(tier_filter)]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", f"{len(filtered_df):,}")
    k2.metric("Return/Cancel Rate", f"{filtered_df['target'].mean()*100:.2f}%")
    k3.metric("Avg Discount %", f"{filtered_df['DiscountPct'].mean()*100:.2f}%")
    k4.metric("Avg Risk Score", f"{filtered_df['risk_score'].mean():.4f}")

    # Chart: risk score distribution
    fig = px.histogram(
        filtered_df, x="risk_score", nbins=50,
        title="Distribution of Risk Scores",
        color_discrete_sequence=["#E63946"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Chart: return rate by category
    cat_summary = filtered_df.groupby("ProductCategory")["target"].mean().sort_values(ascending=False) * 100
    fig2 = px.bar(
        cat_summary, orientation="h",
        title="Return/Cancel Rate by Category (%)",
        labels={"value": "Return/Cancel Rate %", "ProductCategory": ""}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Data table
    st.subheader("Filtered Data")
    st.dataframe(filtered_df.head(500), use_container_width=True)

# ============================================================
# TAB 2: Score a New Order
# ============================================================
with tab2:
    st.header("Score a New Order")
    st.markdown("Enter the details of a new order below to get a live return/cancellation risk score.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer & Product")
        customer_tier = st.selectbox("Customer Tier", categorical_options["CustomerTier"])
        customer_segment = st.selectbox("Customer Segment", categorical_options["CustomerSegment"])
        city_tier = st.selectbox("City Tier", categorical_options["CityTier"])
        product_category = st.selectbox("Product Category", categorical_options["ProductCategory"])

    with col2:
        st.subheader("Order Details")
        channel_type = st.selectbox("Channel Type", categorical_options["ChannelType"])
        payment_method = st.selectbox("Payment Method", categorical_options["PaymentMethod"])
        order_hour = st.slider("Order Hour", 0, 23, 14)
        quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1)
        gross_mrp = st.number_input("Gross MRP Value (₹)", min_value=0.0, value=1500.0, step=100.0)
        discount_pct = st.slider("Discount %", 0.0, 1.0, 0.2, step=0.01)
        is_festival = st.checkbox("Is Festival Period?")
        is_weekend = st.checkbox("Is Weekend?")
        hero_flag = st.checkbox("Is Hero Product?")

    if st.button("🔍 Score This Order", type="primary"):
        input_df = pd.DataFrame([{
            "OrderHour": order_hour,
            "Quantity": quantity,
            "GrossMRPValue": gross_mrp,
            "DiscountPct": discount_pct,
            "IsFestivalPeriod": int(is_festival),
            "IsWeekend": int(is_weekend),
            "HeroFlag": int(hero_flag),
            "CustomerTier": customer_tier,
            "CustomerSegment": customer_segment,
            "CityTier": city_tier,
            "ChannelType": channel_type,
            "PaymentMethod": payment_method,
            "ProductCategory": product_category
        }])

        risk_score = model.predict_proba(input_df)[0, 1]

        if risk_score >= 0.7:
            tier, color = "High Risk", "red"
        elif risk_score >= 0.4:
            tier, color = "Medium Risk", "orange"
        else:
            tier, color = "Low Risk", "green"

        st.markdown("---")
        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Risk Score", f"{risk_score:.4f}")
        c2.markdown(f"### :{color}[{tier}]")

        st.progress(float(risk_score))

        if tier == "High Risk":
            st.warning("⚠️ This order shows elevated risk of return/cancellation. Consider manual review before fulfilment.")
        elif tier == "Medium Risk":
            st.info("ℹ️ This order shows moderate risk. Standard fulfilment recommended, monitor outcome.")
        else:
            st.success("✅ This order shows low risk of return/cancellation.")