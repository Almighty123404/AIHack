# 📊 Credit Default Prediction with Mutual Information & Ensemble Learning

This project builds a robust machine learning pipeline to predict **credit default risk** using advanced feature selection, preprocessing, and ensemble modeling techniques. The primary objective is to maximize **ROC-AUC performance** on a structured financial dataset.

---

## 🚀 Project Overview

This notebook implements an end-to-end workflow for credit risk modeling:

- Data preprocessing & cleaning  
- Feature engineering  
- Mutual Information-based feature selection  
- Model benchmarking  
- Class imbalance handling  
- Ensemble techniques (stacking)  
- Advanced feature transformations  

---

## 🧠 Key Techniques Used

### 🔹 Feature Engineering
- Ordinal encoding for categorical variables  
- Target encoding (advanced stage)  
- Polynomial feature generation  
- Feature interaction exploration  

### 🔹 Feature Selection
- **Mutual Information (MI)** to evaluate feature importance  
- Identification of weak feature-target relationships  

### 🔹 Handling Class Imbalance
- Stratified sampling  
- Model-level adjustments (e.g., class weights)  

### 🔹 Models Implemented
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- HistGradientBoosting (best baseline performer)  
- LightGBM  
- Neural Networks (optional)  

### 🔹 Ensemble Learning
- **Stacking Ensemble (Recommended)**  
  Combines multiple models to improve predictive performance  

---

## 📈 Results Summary

| Version   | Best Model              | ROC-AUC      | Notes                     |
|-----------|------------------------|-------------|--------------------------|
| Baseline  | Multiple               | ~0.59–0.63  | Initial models           |
| Improved  | HistGradientBoosting   | ~0.648      | Better preprocessing     |
| Advanced  | Ensemble Models        | >0.65       | Feature + stacking       |

> **Key Insight:**  
> Performance plateau (~0.62–0.65) suggests **feature limitations**, not model limitations.

---

## ⚠️ Challenges Identified

- Very low Mutual Information scores (~0.012 max)  
- Weak feature-target relationships  
- Limited predictive signal in raw features  
- Class imbalance affecting model performance  

---

## 💡 Improvements Implemented

- Better preprocessing pipeline  
- Class imbalance correction  
- Advanced feature engineering  
- Ensemble stacking for performance boost  

---

## 🏆 Recommended Strategy

### ✅ Best Approach: Stacking Ensemble
- Combines strengths of multiple models  
- Provides the highest potential AUC improvement  

### 🔁 Alternatives
- Advanced feature engineering (target encoding, interactions)  
- LightGBM tuning  

---

## 📂 Project Structure
├── train.ipynb # Main notebook with full pipeline
├── data/ # Dataset (not included)
├── submissions/ # Generated prediction files
└── README.md # Project documentation


---

## ⚙️ Installation

bash
pip install numpy pandas scikit-learn matplotlib seaborn lightgbm

📜 License

This project is open-source and available under the MIT License.
