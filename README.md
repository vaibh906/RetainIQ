# 🚀 RetainIQ
### AI-Powered Customer Intelligence & Churn Prediction System

<p align="center">
  <strong>Understand Customers • Predict Churn • Improve Retention</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Technology Stack](#-technology-stack)
- [Project Workflow](#-project-workflow)
- [Machine Learning Approach](#-machine-learning-approach)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Current Progress](#-current-progress)
- [Future Enhancements](#-future-enhancements)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

## 💡 Overview

**RetainIQ** is a Machine Learning-based customer intelligence system designed to predict customer churn using historical customer data.

Customer churn occurs when customers stop using a company's products or services. It is a major challenge for subscription-based businesses, especially in the telecommunications industry.

RetainIQ aims to identify customers who are more likely to leave a service by analyzing their demographics, subscription details, service usage, and billing information.

The project focuses on building an end-to-end Machine Learning workflow — from data collection and analysis to model training, evaluation, explainability, and deployment.

**Our goal:** Turn customer data into meaningful insights that can help businesses make better retention decisions.

## 🎯 Problem Statement

Businesses often identify customer churn only after customers cancel their subscriptions.

This creates challenges such as:

- Loss of recurring revenue.
- Difficulty identifying customers at risk.
- Limited understanding of customer behavior.
- Missed opportunities for customer retention.

RetainIQ aims to address this challenge by using Machine Learning to predict customer churn based on historical patterns.

## 🎯 Project Objectives

- Understand and analyze customer behavior.
- Identify patterns associated with customer churn.
- Perform data cleaning and preprocessing.
- Build classification models to predict churn.
- Compare multiple Machine Learning algorithms.
- Evaluate models using appropriate performance metrics.
- Understand important factors influencing predictions.
- Explain predictions using SHAP.
- Develop an interactive Streamlit dashboard.

## 📊 Dataset

This project uses the **IBM Telco Customer Churn Dataset**.

**Dataset Source:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)

| Dataset Information | Details |
|---|---|
| Dataset Name | Telco Customer Churn |
| Total Records | 7,043 |
| Total Columns | 21 |
| Problem Type | Binary Classification |
| Target Variable | `Churn` |
| Target Classes | Yes / No |

### Important Features

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `InternetService` | Type of internet service |
| `Contract` | Type of customer contract |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly service charges |
| `TotalCharges` | Total amount charged |
| `Churn` | Whether the customer left the service |

The `Churn` column is the target variable that our Machine Learning model aims to predict.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and Machine Learning |
| XGBoost | Gradient boosting algorithm |
| SHAP | Model explainability |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |
| Git & GitHub | Version control |

## 🔄 Project Workflow

The project follows an end-to-end Machine Learning workflow.

```text
             DATA COLLECTION
                    |
                    v
             DATA UNDERSTANDING
                    |
                    v
              DATA CLEANING
                    |
                    v
        EXPLORATORY DATA ANALYSIS
                    |
                    v
          DATA PREPROCESSING
                    |
                    v
             TRAIN-TEST SPLIT
                    |
                    v
        ENCODING & FEATURE SCALING
                    |
                    v
             MODEL TRAINING
                    |
                    v
            MODEL EVALUATION
                    |
                    v
          HYPERPARAMETER TUNING
                    |
                    v
          MODEL EXPLAINABILITY
                    |
                    v
          STREAMLIT DEPLOYMENT
```

### Workflow Explanation

| Stage | Description |
|---|---|
| Data Collection | Download and load the customer dataset |
| Data Understanding | Explore rows, columns, data types, and features |
| Data Cleaning | Handle missing values and data quality issues |
| EDA | Discover patterns and relationships through analysis |
| Preprocessing | Encode categorical features and scale numerical features |
| Model Training | Train different classification algorithms |
| Evaluation | Compare model performance |
| Explainability | Understand predictions using SHAP |
| Deployment | Build an interactive application using Streamlit |

## 🤖 Machine Learning Approach

RetainIQ is designed as a binary classification problem.

### Planned Algorithms

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Baseline classification model |
| Decision Tree | Rule-based classification |
| Random Forest | Ensemble learning |
| Support Vector Machine | Classification using decision boundaries |
| K-Nearest Neighbors | Similarity-based classification |
| XGBoost | Gradient boosting |

### Model Evaluation Metrics

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

Recall and F1-score will be particularly important when evaluating the model's ability to identify customers who may churn.

## 📁 Project Structure

```text
RetainIQ/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── cleaned_data.csv
│
├── notebooks/
│
├── src/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── check_data_quality.py
│   ├── analyze_target.py
│   ├── analyze_features.py
│   ├── inspect_columns.py
│   ├── clean_data.py
│   ├── analyze_categories.py
│   ├── preprocess_data.py
│   └── split_data.py
│
├── models/
├── reports/
│
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure will evolve as additional modules are developed.*

## ⚙️ Installation and Setup

Follow these steps to set up RetainIQ on your local machine.

### Prerequisites

Make sure you have the following installed:

| Requirement | Recommended Version |
|---|---|
| Python | 3.10 or higher |
| Git | Latest stable version |
| VS Code | Recommended IDE |
| pip | Python package manager |

### Step 1: Clone the Repository

```bash
git clone https://github.com/vaibh906/RetainIQ.git
```

### Step 2: Navigate to the Project Directory

```bash
cd RetainIQ
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Download the Dataset

```bash
python src/download_data.py
```

### Step 7: Clean the Dataset

```bash
python src/clean_data.py
```

### Step 8: Run Preprocessing

```bash
python src/preprocess_data.py
```

**Note:** Run the cleaning script before preprocessing because the preprocessing script uses `data/cleaned_data.csv`.

## 📦 Requirements

All project dependencies are maintained in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

See the [`requirements.txt`](requirements.txt) file for the complete dependency list.

## 📈 Current Progress

<details>
<summary><strong>View development progress</strong></summary>

| Module | Status |
|---|---|
| Dataset Collection | ✅ Completed |
| Data Understanding | ✅ Completed |
| Data Quality Checking | ✅ Completed |
| Data Cleaning | ✅ Completed |
| Target Analysis | ✅ Completed |
| Feature Analysis | ✅ Initial analysis completed |
| Categorical Analysis | ✅ Completed |
| Train-Test Split | ✅ Completed |
| One-Hot Encoding | ✅ Completed |
| Feature Scaling | ✅ Completed |
| Model Training | ⏳ Upcoming |
| Model Evaluation | ⏳ Upcoming |
| Hyperparameter Tuning | ⏳ Upcoming |
| SHAP Explainability | ⏳ Upcoming |
| Streamlit Dashboard | ⏳ Upcoming |

</details>

## 🚀 Future Enhancements

The following features are planned:

- Train and compare multiple Machine Learning algorithms.
- Perform cross-validation.
- Apply hyperparameter tuning.
- Investigate class imbalance handling techniques.
- Improve feature engineering.
- Explain predictions using SHAP.
- Build an interactive Streamlit dashboard.
- Save and load the best-performing model.
- Develop customer-level churn risk analysis.
- Explore customer retention recommendations.

## 🎓 Learning Outcomes

Through this project, the following practical skills are being developed:

- Python programming.
- Data Analysis and EDA.
- Data Cleaning and Preprocessing.
- Feature Engineering.
- Machine Learning Classification.
- Model Evaluation.
- Cross-Validation.
- Hyperparameter Tuning.
- Model Explainability.
- Machine Learning Deployment.

## 👨‍💻 Author

**Vaibhav Pande**

BE — Artificial Intelligence and Data Science

- GitHub: [@vaibh906](https://github.com/vaibh906)
- LinkedIn: [Vaibhav Pande](https://www.linkedin.com/in/vaibhav-pande-996a10360)

---

<p align="center">
  <strong>RetainIQ — Turning Customer Data into Retention Insights.</strong>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>
