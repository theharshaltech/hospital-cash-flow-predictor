import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hospital Cash-Flow Predictor",
    page_icon="🏥",
    layout="wide"
)

AS_OF_DATE = pd.Timestamp("2025-12-31")

st.title("🏥 Hospital Cash-Flow Predictor")
st.write("Insurance claim analysis and payment delay-risk prediction")
st.caption(f"Reporting date: {AS_OF_DATE:%d %b %Y}")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = pd.read_csv("claims.csv")
except FileNotFoundError:
    st.error("claims.csv not found. Run: python generate_data.py")
    st.stop()

required_columns = [
    "claim_id",
    "submission_date",
    "claim_amount",
    "insurance_type",
    "processing_days",
    "status",
    "payment_date"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    st.error(f"Missing required columns: {sorted(missing_columns)}")
    st.stop()

df["submission_date"] = pd.to_datetime(
    df["submission_date"], errors="coerce"
)

df["payment_date"] = pd.to_datetime(
    df["payment_date"], errors="coerce"
)

df["claim_amount"] = pd.to_numeric(
    df["claim_amount"], errors="coerce"
)

df["processing_days"] = pd.to_numeric(
    df["processing_days"], errors="coerce"
)

# Check the core fields before calculating metrics.
if df[["submission_date", "claim_amount", "status"]].isna().any().any():
    st.error(
        "Some claims have missing submission dates, amounts, or statuses. "
        "Please check generate_data.py and regenerate the dataset."
    )
    st.stop()

valid_statuses = {"Paid", "Pending", "Rejected"}

if not set(df["status"].unique()).issubset(valid_statuses):
    st.error("Unexpected claim status found in claims.csv.")
    st.stop()

# --------------------------------------------------
# CLAIM METRICS
# --------------------------------------------------

total_claims = len(df)

paid_claims = df[df["status"] == "Paid"].copy()
pending_claims = df[df["status"] == "Pending"].copy()
rejected_claims = df[df["status"] == "Rejected"].copy()

outstanding_amount = pending_claims["claim_amount"].sum()

# Age of unresolved claims as of the reporting date.
pending_claims["days_open"] = (
    AS_OF_DATE - pending_claims["submission_date"]
).dt.days.clip(lower=0)

# Delay label is meaningful only for paid claims with known
# processing durations.
resolved = paid_claims.dropna(
    subset=["processing_days", "claim_amount", "insurance_type"]
).copy()

# A delay is defined as processing taking more than 30 days.
resolved["delayed"] = (
    resolved["processing_days"] > 30
).astype(int)

# --------------------------------------------------
# CACHED MODEL TRAINING
# --------------------------------------------------

@st.cache_resource
def train_models(data):
    features = ["claim_amount", "insurance_type"]

    X = data[features]
    y = data["delayed"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_features = ["claim_amount"]
    categorical_features = ["insurance_type"]

    # Logistic Regression: scale numeric inputs and encode providers.
    lr_preprocessor = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ])

    logistic_model = Pipeline([
        ("preprocessor", lr_preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Shallow Random Forest for comparison.
    rf_preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ], remainder="passthrough")

    forest_model = Pipeline([
        ("preprocessor", rf_preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    baseline_model = DummyClassifier(strategy="most_frequent")

    models = {
        "Logistic Regression": logistic_model,
        "Shallow Random Forest": forest_model,
        "Majority-Class Baseline": baseline_model
    }

    results = {}

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": estimator,
            "predictions": predictions,
            "probabilities": probabilities,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "pr_auc": average_precision_score(
                y_test, probabilities
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )
        }

    return results, y_test.reset_index(drop=True), X_test.reset_index(drop=True)


if len(resolved) < 20 or resolved["delayed"].nunique() < 2:
    st.error(
        "Not enough paid claims or target-class variation to train and "
        "evaluate the models. Check the dataset before continuing."
    )
    st.stop()

try:
    results, y_test, X_test = train_models(
        resolved[["claim_amount", "insurance_type", "delayed"]]
    )
except ValueError as error:
    st.error(f"Model training failed: {error}")
    st.stop()

# Logistic Regression is the primary model for this demonstration.
primary_model = results["Logistic Regression"]["model"]

# --------------------------------------------------
# FINANCIAL SUMMARY
# --------------------------------------------------

st.divider()
st.subheader("Financial Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Claims", f"{total_claims:,}")
col2.metric("Paid Claims", f"{len(paid_claims):,}")
col3.metric("Pending Claims", f"{len(pending_claims):,}")
col4.metric("Outstanding Amount", f"₹{outstanding_amount:,.0f}")

st.caption(
    "Outstanding amount represents the total value of Pending claims "
    "in this synthetic dataset. It is not a verified forecast of cash "
    "collections."
)

# --------------------------------------------------
# CLAIM STATUS DISTRIBUTION
# --------------------------------------------------

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Claim Status Distribution")

    status_counts = (
        df["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Count")
    )

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Claim Amount by Insurance Provider")

    provider_amounts = (
        df.groupby("insurance_type", as_index=False)["claim_amount"]
        .sum()
    )

    fig = px.bar(
        provider_amounts,
        x="insurance_type",
        y="claim_amount",
        labels={
            "insurance_type": "Insurance Provider",
            "claim_amount": "Total Claim Amount (₹)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PENDING CLAIM AGEING
# --------------------------------------------------

st.divider()
st.subheader("Pending Claim Ageing")

if pending_claims.empty:
    st.info("There are no pending claims in this dataset.")
else:
    pending_claims["Age Group"] = pd.cut(
        pending_claims["days_open"],
        bins=[-1, 15, 30, float("inf")],
        labels=["0–15 days", "16–30 days", "31+ days"]
    )

    ageing = (
        pending_claims["Age Group"]
        .value_counts(sort=False)
        .rename_axis("Age Group")
        .reset_index(name="Claims")
    )

    fig = px.bar(
        ageing,
        x="Age Group",
        y="Claims",
        labels={"Age Group": "Days Open"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Age is calculated from submission date to the fixed reporting "
        "date of 31 Dec 2025."
    )

# --------------------------------------------------
# DELAY RATE BY PROVIDER
# --------------------------------------------------

st.divider()
st.subheader("Observed Delay Rate by Insurance Provider")

if resolved.empty:
    st.info("No paid claims with known processing durations are available.")
else:
    provider_delay = (
        resolved.groupby("insurance_type")
        .agg(
            Paid_Claims=("delayed", "size"),
            Delayed_Claims=("delayed", "sum"),
            Delay_Rate=("delayed", "mean")
        )
        .reset_index()
    )

    provider_delay["Delay_Rate_Percent"] = (
        provider_delay["Delay_Rate"] * 100
    ).round(1)

    fig = px.bar(
        provider_delay,
        x="insurance_type",
        y="Delay_Rate_Percent",
        text="Delay_Rate_Percent",
        labels={
            "insurance_type": "Insurance Provider",
            "Delay_Rate_Percent": "Observed Delay Rate (%)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        provider_delay[
            [
                "insurance_type",
                "Paid_Claims",
                "Delayed_Claims",
                "Delay_Rate_Percent"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Observed rates use paid claims only. Recent unresolved claims "
        "are excluded, which may under-represent slower claims."
    )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.divider()
st.subheader("Machine Learning Model Evaluation")

st.write(
    f"Training and evaluation use {len(resolved):,} paid claims "
    "with known processing durations."
)

comparison_rows = []

for name, result in results.items():
    comparison_rows.append({
        "Model": name,
        "Accuracy": result["accuracy"],
        "Precision": result["precision"],
        "Recall": result["recall"],
        "F1 Score": result["f1"],
        "PR-AUC": result["pr_auc"]
    })

comparison_df = pd.DataFrame(comparison_rows)

metric_df = comparison_df.copy()

for column in ["Accuracy", "Precision", "Recall", "F1 Score", "PR-AUC"]:
    metric_df[column] = metric_df[column].map(
        lambda value: f"{value:.1%}"
    )

st.dataframe(
    metric_df,
    use_container_width=True,
    hide_index=True
)

selected_evaluation_model = st.selectbox(
    "Inspect confusion matrix",
    ["Logistic Regression", "Shallow Random Forest"]
)

matrix = results[selected_evaluation_model]["confusion_matrix"]

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual: Not Delayed", "Actual: Delayed"],
    columns=["Predicted: Not Delayed", "Predicted: Delayed"]
)

st.write(f"Confusion matrix — {selected_evaluation_model}")
st.dataframe(matrix_df, use_container_width=True)

st.caption(
    "Delay means processing took more than 30 days. Accuracy measures "
    "overall correctness; precision measures how often delay predictions "
    "are correct; recall measures how many actual delayed claims are "
    "identified; F1 balances precision and recall; PR-AUC summarizes "
    "precision-recall performance. All results use synthetic data and "
    "a single held-out test split."
)

# --------------------------------------------------
# NEW CLAIM PREDICTION
# --------------------------------------------------

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

    probability = primary_model.predict_proba(new_claim)[0, 1]

    st.metric("Estimated Delay Probability", f"{probability:.1%}")

    st.caption(
        "This is a model score derived from synthetic data, not a "
        "validated probability of a real claim being delayed."
    )

    if probability >= 0.5:
        st.warning(
            "Higher predicted delay risk under this demo model. "
            "Review the claim."
        )
    else:
        st.success(
            "Lower predicted delay risk under this demo model."
        )

# --------------------------------------------------
# CLAIMS EXPLORER
# --------------------------------------------------

st.divider()
st.subheader("Insurance Claims Explorer")

status_filter = st.selectbox(
    "Filter by Claim Status",
    ["All", "Paid", "Pending", "Rejected"]
)

filtered_df = df.copy()

if status_filter != "All":
    filtered_df = filtered_df[
        filtered_df["status"] == status_filter
    ]

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Claims CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="claims_analysis.csv",
    mime="text/csv"
)

# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.divider()
st.subheader("Project Limitations")

st.markdown(
    """
    - This project uses synthetic claims rather than real hospital data.
    - Delay is defined as processing taking more than 30 days.
    - The models use only claim amount and insurance provider.
    - Pending and rejected claims are excluded from model training because
      their completed processing durations are not known.
    - Recent unresolved claims may cause the observed delay rate to
      under-represent slower claims.
    - The model has not been validated for real-world financial decisions.
    """
)