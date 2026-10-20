import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(
    page_title="Hospital Cash-Flow Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Cash-Flow Predictor")
st.write("Insurance claim analysis and payment delay prediction")

# Load dataset
try:
    df = pd.read_csv("claims.csv")
except FileNotFoundError:
    st.error("claims.csv not found. Run python generate_data.py first.")
    st.stop()

df["submission_date"] = pd.to_datetime(df["submission_date"])

# Define a simple, transparent target for this synthetic demo:
# A claim is classified as delayed if processing takes more than 30 days.
# This is a demo threshold, not a real hospital standard.
df["delayed"] = (df["processing_days"] > 30).astype(int)

# Train only on historical claim attributes.
features = ["claim_amount", "insurance_type"]
X = df[features]
y = df["delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("insurance", OneHotEncoder(handle_unknown="ignore"),
         ["insurance_type"])
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Dashboard metrics
total_claims = len(df)
paid_claims = df[df["status"] == "Paid"]
pending_claims = df[df["status"] == "Pending"]
rejected_claims = df[df["status"] == "Rejected"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Claims", f"{total_claims:,}")
col2.metric("Paid Claims", f"{len(paid_claims):,}")
col3.metric("Pending Claims", f"{len(pending_claims):,}")
col4.metric(
    "Outstanding Amount",
    f"₹{pending_claims['claim_amount'].sum():,.0f}"
)

st.divider()

# Visualizations
left, right = st.columns(2)

with left:
    st.subheader("Claim Status Distribution")

    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Claim Amount by Insurance Provider")

    fig = px.bar(
        df.groupby("insurance_type", as_index=False)["claim_amount"].sum(),
        x="insurance_type",
        y="claim_amount",
        labels={
            "insurance_type": "Insurance Provider",
            "claim_amount": "Total Claim Amount (₹)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# Model evaluation
st.divider()
st.subheader("Machine Learning Model Evaluation")

col1, col2 = st.columns(2)

col1.metric(
    "Test Accuracy",
    f"{accuracy_score(y_test, predictions):.1%}"
)

col2.metric(
    "Delayed Claim Recall",
    f"{recall_score(y_test, predictions, zero_division=0):.1%}"
)

st.caption(
    "Synthetic-data demonstration. Delay is defined as processing "
    "time greater than 30 days. Results do not establish real-world "
    "predictive performance."
)

# Prediction form
st.divider()
st.subheader("Predict Delay Risk for a New Claim")

with st.form("prediction_form"):
    amount = st.number_input(
        "Claim Amount (₹)",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=1000
    )

    provider = st.selectbox(
        "Insurance Provider",
        ["Insurance A", "Insurance B", "Insurance C"]
    )

    submitted = st.form_submit_button("Predict Delay Risk")

if submitted:
    new_claim = pd.DataFrame({
        "claim_amount": [amount],
        "insurance_type": [provider]
    })

    probability = model.predict_proba(new_claim)[0][1]

    st.metric("Estimated Delay Probability", f"{probability:.1%}")

    if probability >= 0.5:
        st.warning("Higher predicted delay risk. Review this claim.")
    else:
        st.success("Lower predicted delay risk under this demo model.")

# Claims table
st.divider()
st.subheader("Insurance Claims Explorer")

status_filter = st.selectbox(
    "Filter by Claim Status",
    ["All", "Paid", "Pending", "Rejected"]
)

filtered_df = df.copy()

if status_filter != "All":
    filtered_df = filtered_df[filtered_df["status"] == status_filter]

st.dataframe(filtered_df, use_container_width=True)

st.download_button(
    "Download Claims CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="claims_analysis.csv",
    mime="text/csv"
)