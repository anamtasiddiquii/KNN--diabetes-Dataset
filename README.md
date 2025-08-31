# 🩺 Diabetes Prediction using KNN Imputer + Logistic Regression

This project applies **Machine Learning** to predict diabetes using the famous **Pima Indians Diabetes Dataset**.  
It demonstrates how to handle **missing values** with `KNNImputer` and build a predictive model using **Logistic Regression**.

---

## 📂 Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- Features:
  - Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target:  
  - `Outcome` → **1 = Diabetes**, **0 = No Diabetes**

---

## ⚙️ Workflow
1. **Data Cleaning**
   - Replaced biologically impossible `0` values (in Glucose, Blood Pressure, SkinThickness, Insulin, BMI) with `NaN`.

2. **Missing Value Imputation**
   - Applied **KNNImputer** (`n_neighbors=3`, `weights="distance"`) to fill missing values.

3. **Feature Scaling**
   - Used **StandardScaler** to normalize numerical features for Logistic Regression.

4. **Model Training**
   - Logistic Regression (`max_iter=1000`, `random_state=42`).

5. **Evaluation**
   - Accuracy

---

## 📊 Results
- **Logistic Regression Accuracy:** ~75%  
- Model performs decently, but improvements are possible with:
  - Hyperparameter tuning
  - Alternative models (e.g., Random Forest, Gradient Boosting)
  - Feature engineering

---

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/diabetes-logistic-regression.git
cd diabetes-logistic-regression

# Install dependencies
pip install -r requirements.txt

# Run script
python src/model.py
