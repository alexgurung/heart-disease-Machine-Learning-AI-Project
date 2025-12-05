# Heart Disease Prediction System

**Author:** Alex Gurung  

This project implements a **Heart Disease Prediction System** using machine learning techniques in Python. The system uses the Heart Disease dataset to predict the presence of heart disease based on patient features. Multiple models are trained, tuned, and evaluated for optimal performance.

---

## 🧠 Features & Methods

### 1. Data Preprocessing
- Handling missing values using `SimpleImputer`
- Scaling numeric features with `StandardScaler`
- One-hot encoding categorical features
- Mapping binary columns (e.g., `sex`, `fbs`, `exang`) to 0/1

### 2. Data Splitting
- 60% training, 20% validation, 20% testing
- Stratified split to maintain class balance

### 3. Machine Learning Models
- **Logistic Regression** – Hyperparameter tuning with different `C` values
- **Random Forest Classifier** – Hyperparameter tuning with different `n_estimators`
- **Support Vector Machine (SVM)** – Hyperparameter tuning with different `C` values

### 4. Evaluation
- F1-score, precision, recall, and accuracy
- Best models selected using validation set
- Final unbiased evaluation on the test set

---

## 📊 How to Run

1. Clone the repository:


git clone https://github.com/alexgurung/Recommendation-System.git
cd Recommendation-System
Install Python dependencies (if not already installed):

bash
Copy code
pip install pandas numpy scikit-learn
Place the dataset (heart.csv) in the data/ folder.

Run the main script:

bash
Copy code
python heart_prediction.py
Output includes:

Preprocessed dataset information

Model training logs and hyperparameter tuning results

Evaluation metrics on the test set for all models

🔬 Dataset
Dataset: Heart Disease dataset

Features: Patient demographics, health metrics, and clinical tests

Label: num (presence of heart disease)

📈 Future Enhancements
Implement cross-validation for more robust evaluation

Add feature importance visualization for interpretability

Deploy as a web application using Flask or Streamlit

Include automated hyperparameter tuning with GridSearchCV
