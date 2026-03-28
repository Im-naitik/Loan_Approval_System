# Loan_Approval_System
# 🏦 CreditWise — Intelligent Loan Approval System

A machine learning project built for **SecureTrust Bank** to automate and improve loan approval decisions using historical applicant data.

---

## 📌 Problem Statement

SecureTrust Bank offers personal and home loans to customers across urban and rural regions of India. Their manual loan verification process — where officers review income proofs, employment details, credit history, and other documents — is:

- ⏳ **Time-consuming**
- ⚖️ **Biased & inconsistent**

This leads to two critical problems:
1. **Good customers get rejected** → loss of business
2. **High-risk customers get approved** → financial losses

**Solution:** An intelligent loan approval system powered by Machine Learning that automatically analyses applicant details and predicts whether a loan should be **Approved or Rejected** before final human review.

---

## 📂 Project Structure

```
creditwise/
│
├── credit_wise.ipynb        # Main Jupyter Notebook
├── loan_approval_data.csv   # Dataset
└── README.md
```

---

## 📊 Dataset Description

The dataset contains **1,000 loan applicants** with 20 features covering personal, financial, and credit information.

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique identifier for each applicant |
| `Applicant_Income` | Monthly income of the primary applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Employment_Status` | Salaried / Self-employed |
| `Age` | Applicant's age |
| `Marital_Status` | Single / Married |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit score of the applicant |
| `Existing_Loans` | Number of existing active loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Savings amount |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment term (months) |
| `Loan_Purpose` | Car / Business / Home / Education / Personal |
| `Property_Area` | Urban / Semiurban / Rural |
| `Education_Level` | Education level of applicant |
| `Gender` | Male / Female |
| `Employer_Category` | Government / Private / MNC / Unemployed |
| `Loan_Approved` | **Target variable** — Yes / No |

> **Class distribution:** ~70.2% Rejected, ~29.8% Approved (imbalanced dataset)

---

## 🔄 Project Workflow

### 1. 📥 Data Loading & Exploration
- Loaded CSV with 1,000 rows and 20 columns
- Explored data types, shape, and summary statistics

### 2. 🧹 Data Preprocessing
- Handled **missing values** (5% of data):
  - Numerical columns → filled with **mean** (SimpleImputer)
  - Categorical columns → filled with **most frequent value** (SimpleImputer)

### 3. 📈 Exploratory Data Analysis (EDA)
- Visualised class distribution (pie chart)
- Analysed feature distributions and correlations
- Explored categorical features vs loan approval rates

### 4. 🔠 Feature Engineering & Encoding
- Applied **One-Hot Encoding** to categorical features:
  - `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Created **polynomial/interaction features**:
  - `DTI_Ratio_sq` (DTI Ratio squared)
  - `Credit_Score_sq` (Credit Score squared)

### 5. ⚖️ Feature Scaling
- Applied **StandardScaler** for normalization before model training

### 6. 🤖 Model Training & Evaluation
Multiple classifiers were trained and compared:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 74% | 0.60 | 0.46 | 0.52 |
| K-Nearest Neighbors (k=5) | 74% | 0.60 | 0.46 | 0.52 |
| Naive Bayes (baseline) | 86.5% | 0.80 | 0.74 | 0.77 |
| **Logistic Regression (w/ feature engineering)** | **88%** | **0.78** | **0.84** | **0.81** |
| Naive Bayes (w/ feature engineering) | 85.5% | 0.81 | 0.69 | 0.74 |

> ✅ **Best Model:** Logistic Regression after feature engineering — 88% accuracy with strong recall (0.84), meaning fewer creditworthy applicants are wrongly rejected.

---

## ⚙️ Tech Stack

- **Language:** Python 3.12
- **Environment:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn` — Visualisation
  - `scikit-learn` — ML models, preprocessing, and evaluation

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/your-username/creditwise.git
cd creditwise
jupyter notebook credit_wise.ipynb
```

Make sure `loan_approval_data.csv` is in the same directory as the notebook.

---

## 📉 Evaluation Metrics

Given the class imbalance (70/30 split), accuracy alone is not sufficient. The project evaluates models using:

- **Precision** — Of predicted approvals, how many were correct?
- **Recall** — Of actual approvals, how many were caught?
- **F1 Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN

---

## 🔮 Future Improvements

- Handle class imbalance with SMOTE or class weighting
- Try ensemble models: Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with GridSearchCV
- Build a web interface for loan officers to use the model in production
- Add SHAP/LIME explainability for model decisions

---
