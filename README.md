# Medical Insurance Cost Predictor 🏥

An end-to-end Machine Learning web application built with Python and Streamlit that predicts estimated individual medical insurance costs based on patient demographic and health indicators.

---

## 🛠️ Tech Stack

* **Machine Learning:** Scikit-learn (`RandomForestRegressor`), Pandas
* **Web App Framework:** Streamlit
* **Model Serialization:** Joblib
* **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance & Data Pipeline

* **Algorithm:** Random Forest Regressor (`n_estimators=100`, `random_state=42`)
* **Evaluation Metric:** R² Score of **86.7%** on a 20% test split
* **Data Processing:**
  * Imports patient dataset (`medical-charges.csv`)
  * Encodes categorical attributes (`sex`, `smoker`, `region`) using one-hot encoding (`pd.get_dummies`)
  * Evaluates variance explained using `r2_score` before exporting the trained pipeline to `model.pkl`

---

## ✨ Application Features

* **Interactive User Input:** Allows users to configure age (18–100), BMI (10.0–50.0), number of children (0–5), sex, smoking status, and geographical region.
* **Dynamic Feature Alignment:** Automatically reindexes input vectors to match trained dummy variable columns (`model.feature_names_in_`), filling missing values with `0` to prevent runtime schema mismatches.
* **Real-time Predictions:** Generates immediate formatted cost estimates upon submission.

---

## 🚀 Live Demo

Try the live application here: **[MediPredi](https://github.com/abhinav-kamoji/medical-predictor)**

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhinav-kamoji/medical-predictor.git](https://github.com/abhinav-kamoji/medical-predictor.git)
   cd medical-predictor
