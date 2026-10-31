# Medical Insurance Cost Predictor 🏥

An end-to-end Machine Learning web application that predicts individual medical insurance costs based on user demographics and health indicators. Optimized using **GridSearchCV** hyperparameter tuning and deployed with a responsive UI.

---

## 🛠️ Tech Stack

* **Machine Learning & Pipeline:** Python, Scikit-learn (`RandomForestRegressor`, `GridSearchCV`), Pandas, Joblib
* **Web Application:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## 📊 Model Performance & Optimization

* **Algorithm:** Random Forest Regressor
* **Hyperparameter Tuning:** Automated search via 5-fold cross-validation (`GridSearchCV`) optimizing `n_estimators`, `max_depth`, and `min_samples_split`
* **Performance Metric:** R² Score of **~86.7%**

---

## ✨ Features & Architecture

* **Robust Inference Engine:** Dynamically aligns categorical variables using feature-name reindexing to prevent data shape mismatches.
* **Fault-Tolerant Application:** Built-in exception handling and Streamlit caching (`@st.cache_resource`) for fast and reliable user interactions.
* **Interactive UI:** Simple two-column dashboard for immediate insurance estimate calculations.

---

## 🚀 Live Demo

Try the live application here: **[MediPredi](https://github.com/abhinav-kamoji/medical-predictor)**

---

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhinav-kamoji/medical-predictor.git](https://github.com/abhinav-kamoji/medical-predictor.git)
   cd medical-predictor
