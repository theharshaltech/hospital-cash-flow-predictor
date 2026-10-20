import pandas as pd
import numpy as np

# Reproducible synthetic data
rng = np.random.default_rng(42)

# Dataset configuration
N = 1500
AS_OF_DATE = pd.Timestamp("2025-12-31")

# Generate claim submission dates
submission_dates = (
    pd.Timestamp("2025-01-01")
    + pd.to_timedelta(
        rng.integers(0, 365, N),
        unit="D"
    )
)

# Generate claim amounts
claim_amounts = rng.integers(5000, 200001, N)

# Assign insurance providers
insurance_types = rng.choice(
    ["Insurance A", "Insurance B", "Insurance C"],
    N
)

# Simulate different processing patterns by provider
provider_delay = {
    "Insurance A": 5,
    "Insurance B": 12,
    "Insurance C": 2
}

# Generate simulated processing durations
processing_days = []

for amount, provider in zip(claim_amounts, insurance_types):
    base_delay = 12 + provider_delay[provider]

    if amount > 120000:
        base_delay += 12
    elif amount > 60000:
        base_delay += 5

    delay = round(rng.normal(base_delay, 8))
    processing_days.append(max(3, min(delay, 75)))

processing_days = np.array(processing_days)

# Generate claim outcomes
statuses = []
payment_dates = []
actual_processing_days = []

for i in range(N):

    submission_date = submission_dates[i]
    expected_payment_date = (
        submission_date
        + pd.Timedelta(days=int(processing_days[i]))
    )

    # Reject approximately 5% of claims
    if rng.random() < 0.05:
        status = "Rejected"
        payment_date = pd.NaT
        actual_days = np.nan

    # Claims not resolved by the reporting date remain pending
    elif expected_payment_date > AS_OF_DATE:
        status = "Pending"
        payment_date = pd.NaT
        actual_days = np.nan

    else:
        status = "Paid"
        payment_date = expected_payment_date
        actual_days = processing_days[i]

    statuses.append(status)
    payment_dates.append(payment_date)
    actual_processing_days.append(actual_days)

# Create the dataset
df = pd.DataFrame({
    "claim_id": range(1001, 1001 + N),
    "submission_date": submission_dates,
    "claim_amount": claim_amounts,
    "insurance_type": insurance_types,
    "processing_days": actual_processing_days,
    "status": statuses,
    "payment_date": payment_dates
})

# Save dataset
df.to_csv("claims.csv", index=False)

# Display a summary
print("\nHospital Cash-Flow Dataset Generated Successfully!")
print(f"Total claims: {len(df)}")
print(f"Reporting date: {AS_OF_DATE.date()}")

print("\nClaim Status Distribution:")
print(df["status"].value_counts())

print("\nOutstanding Claim Value:")
outstanding = df.loc[
    df["status"] == "Pending",
    "claim_amount"
].sum()

print(f"Rs. {outstanding:,.2f}")

print("\nAverage Processing Days by Provider (Paid Claims):")
print(
    df[df["status"] == "Paid"]
    .groupby("insurance_type")["processing_days"]
    .mean()
    .round(2)
)

print("\nSaved to claims.csv")