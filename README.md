# 🚗 Used Car Price Prediction using Machine Learning

A machine learning project that predicts the selling price of used cars using multiple regression algorithms. The project follows industry-standard machine learning practices including feature engineering, proper preprocessing, shuffled cross-validation, and model evaluation.

---

## 📸 Dashboard Preview

![ML Dashboard](car_price_ml_dashboard.png)

---

## ✨ Project Highlights

* Built and compared **3 regression models** for used car price prediction.
* Achieved **96.25% R² Score** using **Random Forest Regressor**.
* Engineered domain-specific features such as **Car Age** and **KMs Per Year**.
* Applied proper preprocessing techniques and eliminated **data leakage**.
* Developed a professional **10-panel ML dashboard** for model analysis.
* Implemented **5-Fold Shuffled Cross-Validation** for robust evaluation.

---

## 📊 Dataset

* **Source:** [Kaggle - Car Price Prediction (Used Cars)](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)
* **Records:** 301 Cars
* **Original Features:** 9
* **Missing Values:** None

### Original Features

* Car Name
* Year
* Selling Price
* Present Price
* Driven KMs
* Fuel Type
* Selling Type
* Transmission
* Owner

---
## 📦 Requirements
pandas
numpy
matplotlib
scikit-learn
joblib
jupyter
## 🔧 Project Workflow

### 1️⃣ Data Overview

Loaded and explored the dataset to understand:

* Dataset shape
* Data types
* Missing values
* Sample records

```python
Rows: 301
Columns: 9
No missing values found
```

---

### 2️⃣ Feature Engineering

Created new features to improve predictive performance.

| Feature         | Formula                    | Purpose                  |
| --------------- | -------------------------- | ------------------------ |
| `Car_Age`       | Current Year - Year        | Calculates vehicle age   |
| `KMs_Per_Year`  | Driven_kms / (Car_Age + 1) | Measures usage intensity |
| `Is_Recent_Car` | Car_Age <= 10              | Identifies newer cars    |
| `Brand`         | Extracted from Car_Name    | Captures brand influence |

### Important ML Improvement

* Used `datetime.now().year` to avoid hardcoded values.
* Removed `Depreciation_Ratio` feature because it introduced **target leakage**.

---

### 3️⃣ Data Preprocessing

Applied preprocessing techniques:

* One-Hot Encoding for categorical features:

  * Brand
  * Fuel Type
  * Selling Type
  * Transmission

* Used `drop_first=True` to avoid multicollinearity.

Final dataset:

```python
Feature Matrix Shape : (301, 53)
Target Vector Shape  : (301,)
```

---

### 4️⃣ Train-Test Split

* Training Data: 80%
* Testing Data: 20%
* Random State: 42

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 5️⃣ Model Training & Evaluation

Trained and evaluated three machine learning models.

| Model               | MAE      | RMSE     | R²         | CV R²      |
| ------------------- | -------- | -------- | ---------- | ---------- |
| **Random Forest** ✅ | ₹0.6173L | ₹0.9293L | **0.9625** | 0.9217     |
| Gradient Boosting   | ₹0.5864L | ₹0.9589L | 0.9601     | **0.9262** |
| Linear Regression   | ₹1.1850L | ₹1.7928L | 0.8605     | 0.7866     |

### Cross Validation

Implemented:

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

### Best Model

🏆 **Random Forest Regressor**

```python
R² Score = 0.9625
```

The close relationship between Test R² and Cross-Validation R² indicates strong generalization and minimal overfitting.

---

### 6️⃣ Top Feature Importance

| Feature             | Importance |
| ------------------- | ---------- |
| Present_Price       | 86.46%     |
| Car_Age             | 5.12%      |
| Driven_kms          | 2.47%      |
| KMs_Per_Year        | 1.16%      |
| Transmission_Manual | 0.84%      |
| Brand_land          | 0.79%      |
| Brand_fortuner      | 0.63%      |
| Is_Recent_Car       | 0.44%      |
| Brand_innova        | 0.36%      |
| Brand_city          | 0.34%      |

**Present Price** is the strongest predictor of resale value.

---

### 7️⃣ Visualizations

Developed a professional **10-panel Machine Learning Dashboard** including:

* Selling Price Distribution
* Car Age vs Selling Price
* Average Price by Fuel Type
* Actual vs Predicted Comparison
* Model Performance Comparison
* Top Feature Importance
* Residual Distribution
* Metrics Summary Table

---

### 8️⃣ Sample Predictions

```text
1. ✅ Actual: ₹0.35L | Predicted: ₹0.44L | Error: ₹-0.09L
2. ✅ Actual: ₹10.11L | Predicted: ₹10.55L | Error: ₹-0.44L
3. ✅ Actual: ₹4.95L | Predicted: ₹4.98L | Error: ₹-0.03L
4. ✅ Actual: ₹0.15L | Predicted: ₹0.21L | Error: ₹-0.06L
5. ✅ Actual: ₹6.95L | Predicted: ₹7.64L | Error: ₹-0.69L
6. ✅ Actual: ₹7.45L | Predicted: ₹6.81L | Error: ₹+0.64L
7. ✅ Actual: ₹1.10L | Predicted: ₹1.09L | Error: ₹+0.01L
8. ✅ Actual: ₹0.50L | Predicted: ₹0.60L | Error: ₹-0.10L
```

---

## ✅ ML Best Practices Applied

| Practice                 | Status |
| ------------------------ | ------ |
| Dynamic Year Calculation | ✅      |
| No Data Leakage          | ✅      |
| One-Hot Encoding         | ✅      |
| Cross Validation         | ✅      |
| Shuffled KFold           | ✅      |
| Overfitting Checked      | ✅      |
| Model Serialization      | ✅      |

---

## 🛠️ Tech Stack

| Library      | Purpose               |
| ------------ | --------------------- |
| Pandas       | Data Manipulation     |
| NumPy        | Numerical Computation |
| Matplotlib   | Visualization         |
| Scikit-learn | Machine Learning      |
| Joblib       | Model Saving          |
| Datetime     | Dynamic Date Handling |

---

## 🚀 Setup & Run

### Clone Repository

```bash
git clone https://github.com/anubrata2209/car-price-prediction.git
cd car-price-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset

Download the dataset from Kaggle and place:

```text
car data.csv
```

inside the project folder.

### Run Notebook

```bash
jupyter notebook code.ipynb
```

---

## 📁 Project Structure

```text
car-price-prediction/
│
├── code.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── car_price_ml_dashboard.png
├── best_car_price_model.pkl
└── car data.csv (not tracked)
```

---

## 📌 Key Insights

* Present Price dominates resale prediction.
* Random Forest achieved the best performance.
* Brand information significantly improves prediction quality.
* Eliminating data leakage dramatically improved model reliability.
* Feature engineering increased model effectiveness.

---

## 🔮 Future Improvements

* Deploy using Streamlit.
* Experiment with XGBoost and CatBoost.
* Perform Hyperparameter Tuning.
* Create a real-time prediction web application.

---

## 👤 Author

**Anubrata Parida**

GitHub: https://github.com/anubrata2209
