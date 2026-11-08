# House Price Prediction Using Linear Regression

A Machine Learning project that predicts house prices using Linear Regression and explores how feature engineering and polynomial regression features affect model performance.

## Project Overview

This project implements and compares three Linear Regression models using the King County House Sales dataset.

The objective is to understand how basic property features, engineered features and polynomial transformations influence house price predictions.

The project demonstrates a complete supervised Machine Learning workflow, including data cleaning, feature engineering, feature scaling, model training, cross-validation, evaluation, model comparison and prediction on new house data.

## Dataset

- **Dataset:** King County House Sales
- **File:** `kc_house_data.csv`
- **Source:** [House Sales Dataset – GitHub](https://raw.githubusercontent.com/tanlitung/Datasets/master/kc_house_data.csv)
- **Target Variable:** `price`
- **Data Loading:** Directly from the dataset's GitHub URL using Pandas.

### Selected Dataset Features

| Feature | Description |
|---|---|
| sqft_living | Interior living area |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| yr_built | Year the house was built |
| sqft_lot | Total lot size |
| price | House sale price (target) |

## Technologies & Libraries

- **Python** – Programming language
- **NumPy** – Numerical operations
- **Pandas** – Data cleaning and manipulation
- **Scikit-learn** – Machine Learning, preprocessing, model training and evaluation

## Project Workflow

### 1. Data Loading and Inspection

- Load the dataset using Pandas.
- Inspect dataset dimensions and sample records.
- Examine the selected features and target variable.

### 2. Data Cleaning

- Remove records with zero bedrooms.
- Remove records with zero bathrooms.
- Remove records with zero living area.
- Remove rows with missing values in selected features.

### 3. Feature Engineering

Create additional features to potentially improve predictive performance:

| Feature | Calculation |
|---|---|
| age | 2015 - yr_built |
| lot_size | Renamed from sqft_lot |
| sqft_per_bedroom | sqft / bedrooms |
| bath_to_bed_ratio | bathrooms / bedrooms |
| is_new | 1 if age is less than 5, otherwise 0 |
| is_large | 1 if sqft is greater than 2500, otherwise 0 |

*Note: Property age is calculated using 2015 as the reference year in this implementation.*

### 4. Train-Test Split

- Training data: 80%
- Testing data: 20%
- Random state: 42

### 5. Feature Scaling

Apply StandardScaler to standardize input features before training the regression models.

### 6. Model Training and Evaluation

Train and evaluate three different Linear Regression approaches.

## Models Implemented

### Model 1: Basic Features

Uses four original property features:

- Living area (sqft)
- Bedrooms
- Bathrooms
- Property age

### Model 2: Engineered Features

Extends the basic feature set with additional information:

- Living area
- Bedrooms
- Bathrooms
- Property age
- Lot size
- Square feet per bedroom
- Bathroom-to-bedroom ratio
- New property indicator
- Large property indicator

### Model 3: Polynomial Features

Applies degree-2 PolynomialFeatures transformation to:

- Living area (sqft)
- Property age

This generates additional polynomial and interaction features to capture potential nonlinear relationships.

## Model Evaluation

Each model is evaluated using:

| Metric | Purpose |
|---|---|
| Train R² Score | Measures model fit on training data |
| Test R² Score | Measures predictive performance on unseen data |
| RMSE | Measures average prediction error in the target variable's units |
| Cross-Validation R² | Evaluates model performance across five validation folds |

The project also compares the test R² scores of all three models to determine which feature approach performs better.

## Model Comparison

The project compares:

1. Basic Features
2. Engineered Features
3. Polynomial Features

The model comparison displays:

- Test R² score for each model
- Ranking of models based on R² score
- Change in R² after feature engineering
- Change in R² after polynomial feature transformation

Actual results are generated when the script is executed.

## House Price Prediction

After comparing the models, the project uses the **Engineered Features model** to predict the price of a new house.

### Example House Profile

| Feature | Value |
|---|---:|
| Living Area | 2,200 sqft |
| Bedrooms | 3 |
| Bathrooms | 2 |
| Property Age | 5 years |
| Lot Size | 9,000 sqft |

The program calculates the required engineered features and uses the trained model to generate a predicted house price.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hamzaaumerr/house-price-prediction-linear-regression
cd house-price-prediction-linear-regression
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

```bash
python house_price_prediction.py
```

The dataset is downloaded automatically from its GitHub source when the program runs, so an internet connection is required.