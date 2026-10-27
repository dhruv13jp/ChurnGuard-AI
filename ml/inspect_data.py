
import pandas as pd

# Step 1: Read our CSV file
df = pd.read_csv("data/telco_churn.csv")

# Step 2: Display the first 5 rows
print(df.head())

# Step 3: Display the number of rows and columns
print("Dataset shape:", df.shape)

# Step 4: Display all column names
print("Column names:")
print(df.columns.tolist())

# Step 5: Check data types and missing values
print(df.info())

# Step 6: Count the churn values
print(df["Churn"].value_counts())


# Step 7: Inspect TotalCharges before cleaning
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nNumber of blank TotalCharges values:")
print(df["TotalCharges"].astype(str).str.strip().eq("").sum())

# Step 8: Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("\nTotalCharges data type after conversion:")
print(df["TotalCharges"].dtype)

print("\nMissing TotalCharges values after conversion:")
print(df["TotalCharges"].isna().sum())


# Display the unique categories in important columns
categorical_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))
