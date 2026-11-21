
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression_pipeline(
    X, y, feature_names=None, polynomial_degree=None
):

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Build Pipeline
    steps = []

    # Add polynomial transformation only when requested
    if polynomial_degree is not None:
        steps.append((
            "polynomial",
            PolynomialFeatures(
                degree=polynomial_degree,
                include_bias=False
            )
        ))

    # Scale features and train Linear Regression
    steps.extend([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    model = Pipeline(steps)

    # 3. Train model
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("=== Model Evaluation ===")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2:  {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:,.2f}")

    # 5. Cross-validation
    # Each fold independently fits PolynomialFeatures (if used),
    # StandardScaler, and LinearRegression.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2"
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(
        f"\nCross-validation R2: "
        f"{cv_mean:.4f} (+/- {cv_std:.4f})"
    )

    # 6. Feature importance (regression coefficients)
    if feature_names is not None:

        print("\n=== Feature Importance ===")

        coefficients = model.named_steps["regression"].coef_

        # Get names after polynomial transformation, if applicable
        if polynomial_degree is not None:
            output_feature_names = (
                model.named_steps["polynomial"]
                .get_feature_names_out(feature_names)
            )
        else:
            output_feature_names = feature_names

        importance = pd.DataFrame({
            "feature": output_feature_names,
            "coefficient": coefficients
        }).sort_values(
            "coefficient", key=abs, ascending=False
        )

        print(importance.to_string(index=False))

    return model, cv_mean, test_r2

# 1. LOAD DATASET
url = "https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv"
df = pd.read_csv(url)

print("Original dataset shape:", df.shape)

# 2. DATA CLEANING
df = df[df["bedrooms"] > 0].copy()
df = df[df["bathrooms"] > 0].copy()
df = df[df["sqft_living"] > 0].copy()

df = df.dropna(subset=[
    "sqft_living", "bedrooms", "bathrooms",
    "yr_built", "sqft_lot", "price"
])

print("After cleaning shape:", df.shape)

# 3. FEATURE ENGINEERING
df = df.rename(columns={
    "sqft_living": "sqft",
    "sqft_lot": "lot_size"
})

df["age"] = 2015 - df["yr_built"]

df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]
df["bath_to_bed_ratio"] = df["bathrooms"] / df["bedrooms"]

df["is_new"] = (df["age"] < 5).astype(int)
df["is_large"] = (df["sqft"] > 2500).astype(int)

assert np.isfinite(
    df[["sqft_per_bedroom", "bath_to_bed_ratio"]].values
).all()

print("\nDataset After Feature Engineering:")
print(df[[
    "sqft", "bedrooms", "bathrooms", "age", "lot_size",
    "sqft_per_bedroom", "bath_to_bed_ratio",
    "is_new", "is_large", "price"
]].head())

y = df["price"].values

# 4. MODEL 1: BASIC FEATURES
print("\n========== MODEL 1: BASIC FEATURES ==========")

basic_features = [
    "sqft", "bedrooms", "bathrooms", "age"
]

X_basic = df[basic_features].values

model_basic, cv_basic, r2_basic = linear_regression_pipeline(
    X_basic,
    y,
    feature_names=basic_features
)

# 5. MODEL 2: ENGINEERED FEATURES
print("\n========== MODEL 2: ENGINEERED FEATURES ==========")

engineered_features = [
    "sqft", "bedrooms", "bathrooms", "age", "lot_size",
    "sqft_per_bedroom", "bath_to_bed_ratio", "is_new", "is_large"
]

X_engineered = df[engineered_features].values

model_eng, cv_eng, r2_eng = linear_regression_pipeline(
    X_engineered,
    y,
    feature_names=engineered_features
)

# 6. MODEL 3: POLYNOMIAL FEATURES
print("\n========== MODEL 3: POLYNOMIAL FEATURES ==========")

# Pass ORIGINAL features into the function.
# PolynomialFeatures will be applied inside the Pipeline.

poly_features = ["sqft", "age"]

X_poly_original = df[poly_features].values

model_poly, cv_poly, r2_poly = linear_regression_pipeline(
    X_poly_original,
    y,
    feature_names=poly_features,
    polynomial_degree=2
)

print("\nPolynomial feature names:")
print(
    model_poly.named_steps["polynomial"]
    .get_feature_names_out(poly_features)
)

# 7. MODEL COMPARISON
print("\n========== MODEL COMPARISON ==========")

results = pd.DataFrame({
    "Model": [
        "Basic Features",
        "Engineered Features",
        "Polynomial Features"
    ],
    "CV R2 Score": [
        cv_basic,
        cv_eng,
        cv_poly
    ],
    "Test R2 Score": [
        r2_basic,
        r2_eng,
        r2_poly
    ]
})

# Rank models using cross-validation, not test performance
print(
    results.sort_values(
        by="CV R2 Score",
        ascending=False
    ).to_string(index=False)
)

print(
    f"\nCV R2 change after Feature Engineering: "
    f"{(cv_eng - cv_basic) * 100:.2f} percentage points"
)

print(
    f"CV R2 change after Polynomial Features: "
    f"{(cv_poly - cv_basic) * 100:.2f} percentage points"
)

# 8. PREDICT A NEW HOUSE PRICE
new_house = pd.DataFrame({
    "sqft": [2200],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [5],
    "lot_size": [9000],
    "sqft_per_bedroom": [2200 / 3],
    "bath_to_bed_ratio": [2 / 3],
    "is_new": [0],
    "is_large": [0]
})

# Pipeline automatically scales the input features
predicted_price = model_eng.predict(
    new_house[engineered_features].values
)[0]

print("\n========== NEW HOUSE PREDICTION ==========")
print("Features: 2,200 sqft | 3 bed | 2 bath | 5 yrs old | 9,000 sqft lot")
print(f"Predicted House Price: ${predicted_price:,.2f}")