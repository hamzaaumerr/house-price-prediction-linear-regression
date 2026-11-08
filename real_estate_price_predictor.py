import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_pipeline(X, y, feature_names=None):

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    print("=== Model Evaluation ===")
    print(f"Train R2: {r2_score(y_train, y_pred_train):.4f}")
    print(f"Test R2:  {r2_score(y_test, y_pred_test):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")

    # 5. Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Feature importance
    if feature_names:
        print("\n=== Feature Importance ===")
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        print(importance.to_string(index=False))

    return model, scaler

url = "https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv"
df = pd.read_csv(url)

print("Original dataset shape:", df.shape)

# Clean rows where bedrooms / bathrooms / sqft are zero or missing,
df = df[df["bedrooms"] > 0].copy()
df = df[df["bathrooms"] > 0].copy()
df = df[df["sqft_living"] > 0].copy()
df = df.dropna(subset=["sqft_living", "bedrooms", "bathrooms",
                       "yr_built", "sqft_lot", "price"])

print("After cleaning shape:", df.shape)

df = df.rename(columns={"sqft_living": "sqft", "sqft_lot": "lot_size"})
df["age"] = 2015 - df["yr_built"]

df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]
df["bath_to_bed_ratio"] = df["bathrooms"] / df["bedrooms"]
df["is_new"] = (df["age"] < 5).astype(int)
df["is_large"] = (df["sqft"] > 2500).astype(int)

assert np.isfinite(df[["sqft_per_bedroom", "bath_to_bed_ratio"]].values).all()

print("\nDataset After Feature Engineering:")
print(df[["sqft", "bedrooms", "bathrooms", "age", "lot_size",
          "sqft_per_bedroom", "bath_to_bed_ratio", "is_new", "is_large", "price"]].head())

y = df["price"].values

print("\n========== MODEL 1: BASIC FEATURES ==========")
basic_features = ["sqft", "bedrooms", "bathrooms", "age"]
X_basic = df[basic_features].values
model_basic, scaler_basic = linear_regression_pipeline(
    X_basic, y, feature_names=basic_features
)

print("\n========== MODEL 2: ENGINEERED FEATURES ==========")
engineered_features = [
    "sqft", "bedrooms", "bathrooms", "age", "lot_size",
    "sqft_per_bedroom", "bath_to_bed_ratio", "is_new", "is_large",
]
X_engineered = df[engineered_features].values
model_eng, scaler_eng = linear_regression_pipeline(
    X_engineered, y, feature_names=engineered_features
)

print("\n========== MODEL 3: POLYNOMIAL FEATURES ==========")
X_poly_original = df[["sqft", "age"]].values
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_poly_original)

print("\nOriginal feature names:")
print(["sqft", "age"])
print("\nPolynomial feature names:")
print(poly.get_feature_names_out(["sqft", "age"]))

model_poly, scaler_poly = linear_regression_pipeline(
    X_poly, y, feature_names=list(poly.get_feature_names_out(["sqft", "age"]))
)

print("\n========== MODEL COMPARISON ==========")

def compute_test_r2(X, y, scaler, model):
    """Reproduce the same split used in training and apply the fitted scaler."""
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return r2_score(y_test, model.predict(scaler.transform(X_test)))


r2_basic = compute_test_r2(X_basic, y, scaler_basic, model_basic)
r2_eng   = compute_test_r2(X_engineered, y, scaler_eng, model_eng)
r2_poly  = compute_test_r2(X_poly, y, scaler_poly, model_poly)

results = pd.DataFrame({
    "Model": ["Basic Features", "Engineered Features", "Polynomial Features"],
    "R2 Score": [r2_basic, r2_eng, r2_poly],
})

print(results.sort_values(by="R2 Score", ascending=False).to_string(index=False))
print(f"\nImprovement from Feature Engineering: {(r2_eng - r2_basic) * 100:.2f}%")
print(f"Improvement from Polynomial Features: {(r2_poly - r2_basic) * 100:.2f}%")

# PREDICT A NEW HOUSE PRICE
new_house = pd.DataFrame({
    "sqft": [2200], "bedrooms": [3], "bathrooms": [2], "age": [5],
    "lot_size": [9000], "sqft_per_bedroom": [2200 / 3],
    "bath_to_bed_ratio": [2 / 3], "is_new": [0], "is_large": [0],
})

new_house_scaled = scaler_eng.transform(new_house[engineered_features].values)
predicted_price = model_eng.predict(new_house_scaled)[0]

print("\n========== NEW HOUSE PREDICTION ==========")
print(f"Features: 2,200 sqft | 3 bed | 2 bath | 5 yrs old | 9,000 sqft lot")
print(f"Predicted House Price: ${predicted_price:,.2f}")