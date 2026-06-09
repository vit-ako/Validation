# Rent Price Prediction: Cross-Validation, Feature Selection & Hyperparameter Optimization

## 📋 Overview

This project is a comprehensive machine learning study focused on predicting rental apartment prices. It implements and compares:

- **5 custom cross-validation methods** (KFold, Stratified, Grouped, Time Series) against scikit-learn implementations
- **3 feature selection techniques** (Lasso, Permutation Importance, SHAP)
- **2 hyperparameter optimization approaches** (Grid Search vs Bayesian Optimization with Optuna)

The project demonstrates how different validation strategies affect model performance and which features most strongly influence rental prices.


## 🔧 Data Preprocessing

- Outlier removal using 1st and 99th percentiles on price
- Binary feature creation from top 20 most frequent amenities (e.g., Elevator, Doorman, Dishwasher)
- Feature set: 23 features (bathrooms, bedrooms, interest_level + 20 amenities)
- Target variable: `price` (rental price in USD)

## 🧪 Implemented Methods

### 1. Data Splitting Methods

| Function | Description |
|----------|-------------|
| `split_test_train()` | Random split (train/test) |
| `split_data()` | Random split (train/val/test) |
| `split_train_test_date()` | Time-based split (train/test) |
| `split_by_date()` | Time-based split (train/val/test) |

### 2. Cross-Validation Methods (Custom vs sklearn)

| Custom Method | sklearn Equivalent | Key Feature |
|---------------|-------------------|-------------|
| `my_KFold()` | `KFold` | Shuffling control |
| `grouped_KFold()` | `GroupKFold` | Group by bathrooms/bedrooms |
| `stratified_Kfold()` | `StratifiedKFold` | Stratify by price quantiles |
| `time_KFold()` | `TimeSeriesSplit` | Temporal ordering |

### 3. Feature Selection Methods

| Method | Type | Description |
|--------|------|-------------|
| **Lasso (L1)** | Embedded | Zeroes out unimportant features |
| **Permutation Importance** | Model-agnostic | Measures performance drop when shuffling |
| **SHAP** | Game-theoretic | Explains individual predictions |

### 4. Hyperparameter Optimization

- **GridSearchCV**: Exhaustive search over 36 combinations
- **Optuna**: Bayesian optimization with 20 trials (with/without K-Fold)

## 📊 Results

### Cross-Validation Comparison

**Metric**: Average absolute difference between train and test mean prices

| Method | Avg Mean Diff | Std | Status |
|--------|---------------|-----|--------|
| **Custom Stratified** | **3.86** ✓ | 2.12 | WORKING |
| Sklearn Stratified | 4.20 | 2.29 | WORKING |
| Custom KFold | 11.57 | 6.43 | WORKING |
| Sklearn KFold | 11.44 | 6.65 | WORKING |
| Custom Time | 39.75 | 36.24 | WORKING |
| Sklearn Time | 39.84 | 36.26 | WORKING |
| Custom Grouped | 1175.86 | 788.84 | WORKING |
| Sklearn Grouped | 1211.08 | 601.39 | WORKING |

> **🏆 Best Method**: `Custom_Stratified` — minimizes distribution shift between folds

### Feature Selection Results (Lasso Model)

| Model | Train MAE | Valid MAE | Test MAE | Test RMSE | Test R² |
|-------|-----------|-----------|----------|-----------|---------|
| Lasso (all 23 features) | 683.88 | 694.99 | 698.16 | 1016.88 | **0.606** |
| Lasso + top-10 (Lasso) | 683.88 | 694.99 | 698.16 | 1016.88 | **0.606** |
| Lasso + top-10 (Permutation) | 687.57 | 698.14 | 702.87 | 1023.27 | 0.601 |
| Lasso + top-10 (SHAP) | 687.57 | 698.14 | 702.87 | 1023.27 | 0.601 |

### Top-10 Most Important Features

| Rank | Lasso Coefficient | Permutation Importance | SHAP Values |
|------|-------------------|------------------------|-------------|
| 1 | bathrooms | bathrooms | bathrooms |
| 2 | bedrooms | bedrooms | bedrooms |
| 3 | interest_level | Doorman | Doorman |
| 4 | Doorman | interest_level | interest_level |
| 5 | LaundryinUnit | LaundryinUnit | LaundryinUnit |
| 6 | Elevator | Elevator | Elevator |
| 7 | LaundryInBuilding | FitnessCenter | LaundryInBuilding |
| 8 | HighSpeedInternet | LaundryinBuilding | FitnessCenter |
| 9 | FitnessCenter | Dishwasher | Dishwasher |
| 10 | LaundryinBuilding | HardwoodFloors | HardwoodFloors |

**Key insight**: All three methods converge on the same top features, confirming their importance.

### Hyperparameter Optimization (ElasticNet)

| Method | Trials | Best Params | Best MSE | Validation RMSE | R² |
|--------|--------|-------------|----------|-----------------|-----|
| Grid Search | 108 | `{alpha:0.01, l1_ratio:1.0}` | 997,287 | 1015.38 | 0.6023 |
| Optuna (no KFold) | 20 | `{alpha:0.01, l1_ratio:1.0}` | 1,030,996 | - | - |
| Optuna (with KFold) | 20 | `{alpha:0.01, l1_ratio:1.0}` | 996,409 | - | - |

> **Efficiency gain**: Optuna achieves comparable results to Grid Search using **5.4x fewer trials** (20 vs 108)

## 💡 Key Findings

1. **Stratified cross-validation** is crucial for price prediction tasks to maintain price distribution across folds.

2. **Lasso** performs as well as more complex methods (SHAP, Permutation) for feature selection on this dataset, while being computationally cheaper.

3. **Top price predictors**:
   - Number of bathrooms (strongest)
   - Number of bedrooms
   - Interest level (high/medium/low)
   - Building amenities (Doorman, Elevator, Laundry)

4. **Bayesian optimization** (Optuna) achieves Grid Search quality with significantly fewer evaluations.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rent-price-prediction.git
cd rent-price-prediction

# Install dependencies
pip install -r requirements.txt
