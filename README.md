# Loan Default Prediction in P2P Lending  
**Integrating Explainable AI into Ensemble Machine Learning Models for Enhanced Credit Risk Prediction**

This project builds and evaluates a set of machine learning models to predict **loan default risk** in a peer-to-peer (P2P) lending context.  
It focuses on:

- Reducing **false negatives** (i.e. missing risky borrowers),
- Evaluating **business cost impact** of misclassifications, and
- Improving transparency using **Explainable AI (XAI)** with **LIME**.

All of the work is implemented in the Jupyter notebook:

> `P2P_Bondara.ipynb`

---

## 📌 1. Project Overview

P2P lending platforms need to assess whether a borrower will **default** or **repay**.  
This notebook walks through the full workflow:

1. Data loading and cleaning  
2. Feature engineering & selection  
3. Model training and hyperparameter tuning  
4. Threshold tuning to minimise **false negatives**  
5. Business cost analysis of prediction errors  
6. Model explainability using **LIME** for local interpretations

The final output is a tuned classification model optimised for **recall**, plus XAI explanations that show **why** the model classifies a specific borrower as “default” or “no default”.

---

## 💾 2. Data

The dataset represents historical loan records from a P2P lending platform.  
Each row corresponds to a loan and contains:

- **Borrower information** (e.g. income-related fields, employment, etc.)
- **Loan characteristics** (e.g. amount, monthly payment, maturity)
- **Account / behavioural information**
- **Outcome variable** indicating default / non-default

Key steps applied in the notebook:

- Removal of **obsolete / leakage features**, including:
  - IDs and purely technical columns (e.g. `LoanId`, `LoanNumber`)
  - Several income breakdown fields
- Removal of multiple **date-related columns** that are not useful for modelling (e.g. `ListedOnUTC`, `BiddingStartedOn`, `LoanApplicationStartedDate`, etc.)
- Creation of a **binary target variable**:
  - Original status labels like *Repaid*, *Current*, *Late* and a `DefaultDate` field are used.
  - Status + default date information are combined into a single target column: `LoanStatus` (default vs no default).

> 📌 You will need the loan dataset (e.g. `LoanData.csv`) stored in your own environment.  
> The notebook expects it to be loaded from Google Drive – update the path if running locally.

---

## 🧹 3. Data Preparation

The data preparation pipeline in the notebook includes:

1. **Missing Value Handling**  
   - Identification of missing values per column.  
   - Appropriate imputation or column removal depending on missingness and relevance.

2. **Outlier Handling**  
   - Use of visualisations (boxplots, histograms) to inspect skewness and extreme values.  
   - Outlier treatment using distribution-based rules (e.g. IQR) for key numeric features such as amount and monthly payment.

3. **Type & Category Cleanup**  
   - Ensuring numeric vs categorical types are correct.
   - Converting selected integer/float columns into categorical where appropriate.

4. **Encoding Categorical Variables**  
   - **Label Encoding** is applied:
     - Target (`y = LoanStatus`) is encoded to 0/1.
     - All categorical predictors in `X` are iterated and label-encoded.

5. **Feature Scaling**  
   - **StandardScaler** is used to scale `X` after encoding.

6. **Feature Selection**  
   - `SelectKBest` with `mutual_info_classif` is applied to select the **top 15 features** most relevant to the default prediction target.

7. **Train–Test Split**  
   - Data is split into training and test sets for unbiased evaluation.

---

## 🤖 4. Models & Algorithms

The notebook trains and compares several classification models:

- **Logistic Regression**
- **Linear Support Vector Machine (LinearSVC)**
- **Random Forest Classifier**
- **XGBoost Classifier**

A common training pattern is used:

- Hyperparameters defined in a dictionary for each model.
- **GridSearchCV** with cross-validation to:
  - Tune hyperparameters,
  - Evaluate model performance using multiple metrics.

For each model, the notebook collects:

- Best hyperparameters,
- Validation scores: **precision**, **recall**, **ROC AUC**, and **F1-score**.

A comparison plot is generated to visualise how each model performs across these metrics.

---

## 🎯 5. Model Selection & Threshold Tuning

Instead of simply picking the model with the best overall F1-score, the project focuses on **minimising False Negatives (FN)** — i.e. borrowers who default but are incorrectly predicted as safe.

Steps:

1. **Model Selection**  
   - From the GridSearch results, the **best model is selected based on Recall**, not just accuracy or F1.

2. **Threshold Tuning**  
   - The chosen model outputs class **probabilities** for the “default” class.
   - A range of thresholds (e.g. 0.1 → 0.5) is tested.
   - For each threshold:
     - Predictions are made,
     - Recall is calculated.
   - The threshold that yields the **highest recall** is selected as the **best operating point**.

This is crucial for credit risk, where missing a risky borrower (FN) is often far more costly than incorrectly flagging a good one (FP).

---

## 📊 6. Evaluation & Metrics

The notebook calculates and visualises several metrics:

- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC Curve & AUC**

The confusion matrix is particularly important, as it underpins the **business cost analysis**.

---

## 💶 7. Business Cost Analysis

The notebook explicitly quantifies the financial impact of wrong predictions:

- Assumed costs:
  - `FN_cost = 5000` → Cost of granting a loan to a borrower who actually defaults (False Negative).
  - `FP_cost = 500`  → Cost of incorrectly rejecting or flagging a good borrower (False Positive).

Using the confusion matrix:

- `fn_count = cm[1, 0]`
- `fp_count = cm[0, 1]`

The total cost is computed as:

```text
Total Misclassification Cost
= FN_cost × FN_count + FP_cost × FP_count

A bar chart then visualises:
	•	Total cost from False Negatives vs
	•	Total cost from False Positives

This ties the model performance back to real business impact.

⸻

🧠 8. Explainable AI with LIME

To improve transparency, the project uses LIME (Local Interpretable Model-Agnostic Explanations):
	•	A LimeTabularExplainer is created using the scaled training data and the original feature names.
	•	A specific test instance (e.g. the first row of X_test) is selected.
	•	LIME generates:
	•	A local explanation showing the top features that pushed the model towards “Default” or “No Default”.
	•	These are displayed in-notebook using exp.show_in_notebook(...).

This allows stakeholders to understand why the model predicted that a given borrower is risky, which is critical for:
	•	Regulatory compliance,
	•	Internal risk governance,
	•	Customer communication.

⸻

📁 9. Project Structure

A typical repository layout for this project could look like:

.
├── P2P_Bondara.ipynb      # Main notebook with full pipeline
├── LoanData.csv           # (Not included) Raw loan dataset – add your own
├── README.md              # Project documentation (this file)
└── requirements.txt       # Python dependencies (optional)

Note: The dataset is not bundled here. You must provide your own loan dataset and update the path in the notebook.

⸻

⚙️ 10. How to Run the Notebook

Option A – Google Colab (recommended)
	1.	Upload P2P_Bondara.ipynb to Google Colab.
	2.	Upload the dataset (e.g. LoanData.csv) to your Google Drive.
	3.	In the notebook:
	•	The following lines mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


	•	Update the pd.read_csv(...) path to point to your dataset location in Drive.

	4.	Run all cells, top to bottom.

Option B – Local Jupyter
	1.	Install dependencies (example):

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lime


	2.	Place LoanData.csv in a local folder and update the file path in the read_csv call.
	3.	Launch Jupyter:

jupyter notebook


	4.	Open P2P_Bondara.ipynb and run all cells.

⸻

📦 11. Dependencies

Core Python libraries used:
	•	pandas
	•	numpy
	•	matplotlib
	•	seaborn
	•	scikit-learn
	•	LogisticRegression, LinearSVC, RandomForestClassifier
	•	train_test_split, KFold, GridSearchCV, ShuffleSplit
	•	StandardScaler, SelectKBest, mutual_info_classif
	•	Metrics: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
	•	xgboost (XGBClassifier)
	•	lime (LimeTabularExplainer)

You can optionally create a requirements.txt with these packages for reproducibility.

⸻

🚀 12. Possible Extensions

Some ideas to take this further:
	•	Try additional models or stacked ensembles.
	•	Apply class weighting or cost-sensitive learning directly in the algorithms.
	•	Use SMOTE or other resampling methods if the dataset is highly imbalanced.
	•	Extend XAI:
	•	Global explanations (e.g. feature importance, SHAP values),
	•	Compare human-interpretable rules to model outputs.
	•	Deploy the best model as an API or simple web dashboard for risk analysts.

⸻

🙏 13. Acknowledgements

This project builds on public P2P lending datasets and standard Python ML/XAI libraries.
Special thanks to the open-source community behind scikit-learn, XGBoost, and LIME, and to the creators of the underlying loan dataset.
