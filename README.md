# 🏦 Bank Customer Churn Prediction

A machine learning project to predict customer churn for a bank using R. The project covers full pipeline: exploratory data analysis, handling class imbalance with SMOTE, and comparing three classification models.

---

## 📁 Project Structure

```
bank-churn-analysis/
│
├── churn_analysis.R              # Full R script
├── Customer-Churn-Records.csv    # Dataset
├── roc_comparison.png            # ROC curve comparison plot
└── README.md
```

---

## 📊 Dataset

**Customer-Churn-Records.csv** — Bank customer data with 10,000 records.

| Feature | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Has credit card (0/1) |
| `IsActiveMember` | Active member (0/1) |
| `EstimatedSalary` | Estimated annual salary |
| `Satisfaction.Score` | Customer satisfaction score |
| `Card.Type` | Type of card (Silver/Gold/Platinum) |
| `Point.Earned` | Loyalty points earned |
| `Exited` | **Target** — churned (1) or stayed (0) |

> **Dropped columns:** `RowNumber`, `CustomerId`, `Surname` (identifiers), `Complain` (data leakage)

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis
- Churn distribution
- Churn by Geography, Gender, Age, Balance
- Credit Score, NumOfProducts, Satisfaction Score analysis

### 2. Handling Class Imbalance
- Original distribution: **80% stayed / 20% churned**
- Applied **SMOTENC** (handles mixed numeric + factor data) via `themis` package
- After SMOTE: **50/50 balanced classes**

```
Before SMOTE:   0: 6370  |  1: 1631
After SMOTE:    0: 6370  |  1: 6370
```

### 3. Models Trained
- Logistic Regression (baseline)
- Random Forest
- XGBoost (basic + tuned with random search)

### 4. Evaluation
- 5-fold Cross Validation
- Metrics: AUC, Sensitivity, Specificity, Precision, Kappa
- ROC curve comparison across all models

---

## 📈 Results

| Model | AUC | Sensitivity | Specificity | Balanced Accuracy | Kappa |
|---|---|---|---|---|---|
| Logistic Regression | 0.778 | 69.3% | 71.9% | 70.6% | 0.318 |
| Random Forest | 0.862 | 61.7% | 90.5% | 76.1% | 0.523 |
| XGBoost Basic | 0.869 | 53.8% | 93.2% | 73.5% | 0.506 |
| XGBoost Tuned | 0.857 | 55.3% | 92.4% | 73.8% | 0.505 |

![ROC Curve Comparison](roc_comparison.png)

### 🏆 Best Model: Random Forest
- Best Kappa (0.523) and Balanced Accuracy (76.1%)
- No overfitting — consistent CV and test performance
- Most interpretable for business stakeholders

---

## 🌟 Key Findings

1. **Age** is the strongest predictor of churn by far
2. **NumOfProducts** and **Balance** are the next most important features
3. **Card type** and **Geography** have minimal predictive power
4. **SMOTE was critical** — without it, sensitivity dropped from 69% to just 22%
5. XGBoost showed overfitting signs (CV AUC: 0.964 vs test AUC: 0.857)

---

## 🛠️ Libraries Used

```r
library(tidyverse)    # data manipulation
library(ggplot2)      # visualisation
library(themis)       # SMOTE for class imbalance
library(caret)        # model training framework
library(randomForest) # random forest
library(xgboost)      # XGBoost
library(pROC)         # ROC curves and AUC
```

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/bank-churn-analysis.git
cd bank-churn-analysis
```

2. Install required R packages
```r
install.packages(c("tidyverse", "ggplot2", "themis", "caret", 
                   "randomForest", "xgboost", "pROC"))
```

3. Run the analysis
```r
source("churn_analysis.R")
```

---

