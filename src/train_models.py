import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 1. Load the dataset
data = pd.read_csv('data/heart.csv')


# 2. Separate features (X) and labels (y)
# Drop columns not used for prediction
X = data.drop(['num', 'id', 'dataset'], axis=1)
y = data['num']

# 3. Split data into train, validation, and test sets
# First, split off 20% for test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# Then split train+val into 75% train, 25% val (so final: 60% train, 20% val, 20% test)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


# Show class distribution and shapes
print("Class distribution in training set:")
print(y_train.value_counts())
print("\nClass distribution in validation set:")
print(y_val.value_counts())
print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")


# 4. Map binary columns to 0/1
binary_map = {'Male': 1, 'Female': 0, True: 1, False: 0}
binary_cols = ['sex', 'fbs', 'exang']
for df in [X_train, X_val, X_test]:
    df[binary_cols] = df[binary_cols].replace(binary_map)

# 5. Preprocessing: one-hot encode categorical, scale numeric, impute missing
multi_cat_cols = ['cp', 'restecg', 'slope', 'thal']
numerical_cols = [col for col in X.columns if col not in binary_cols + multi_cat_cols]

# ColumnTransformer: one-hot for categorical, scale for numeric, passthrough for binary
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), multi_cat_cols),
    ('scaler', StandardScaler(), numerical_cols)
], remainder='passthrough')

# Impute missing values (mean strategy)
imputer = SimpleImputer(strategy='mean')
X_train_pre = imputer.fit_transform(preprocessor.fit_transform(X_train))
X_val_pre = imputer.transform(preprocessor.transform(X_val))
X_test_pre = imputer.transform(preprocessor.transform(X_test))

print(f"Preprocessed train data shape: {X_train_pre.shape}")
print(f"Preprocessed val data shape: {X_val_pre.shape}")
print(f"Preprocessed test data shape: {X_test_pre.shape}")

# Check for missing values after imputation
print("Missing values in train:", np.isnan(X_train_pre).sum())
print("Missing values in val:", np.isnan(X_val_pre).sum())
print("Missing values in test:", np.isnan(X_test_pre).sum())


# 6. Logistic Regression: hyperparameter tuning (C)
param_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
best_f1 = 0
best_C = None
for c in param_grid:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_pre, y_train)
    y_pred = model.predict(X_val_pre)
    print(f"C={c}:")
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))
    f1 = f1_score(y_val, y_pred, average='weighted')
    if f1 > best_f1:
        best_f1 = f1
        best_C = c
        best_model = model
print(f"\nBest C: {best_C} with F1-score: {best_f1:.4f}")


# 7. Random Forest: hyperparameter tuning (n_estimators)
rf_param_grid = [50, 100, 150, 200, 250]
best_rf_f1 = 0
best_rf_n = None
for n in rf_param_grid:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, class_weight='balanced')
    rf.fit(X_train_pre, y_train)
    y_val_pred = rf.predict(X_val_pre)
    print(f"Random Forest - n_estimators={n}:")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    if f1 > best_rf_f1:
        best_rf_f1 = f1
        best_rf_n = n
        best_rf_model = rf
print(f"\nBest Random Forest n_estimators: {best_rf_n} with F1-score: {best_rf_f1:.4f}")


# 8. SVM: hyperparameter tuning (C)
svm_param_grid = [0.1, 1, 10, 100, 1000]
best_svm_f1 = 0
best_svm_C = None
for c in svm_param_grid:
    svm = SVC(C=c, kernel='rbf', class_weight='balanced', gamma='scale', random_state=42)
    svm.fit(X_train_pre, y_train)
    y_val_pred = svm.predict(X_val_pre)
    print(f"SVM - C={c}:")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    if f1 > best_svm_f1:
        best_svm_f1 = f1
        best_svm_C = c
        best_svm_model = svm
print(f"\nBest SVM C: {best_svm_C} with F1-score: {best_svm_f1:.4f}")


# 9. Evaluate best models on test set (final unbiased evaluation)
print("\nTesting Best Models on Test Set")

# Logistic Regression
print("\nLogistic Regression test results:")
best_model.fit(np.vstack([X_train_pre, X_val_pre]), pd.concat([y_train, y_val]))
y_test_pred = best_model.predict(X_test_pre)
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

# Random Forest
print("\nRandom Forest test results:")
best_rf_model.fit(np.vstack([X_train_pre, X_val_pre]), pd.concat([y_train, y_val]))
y_test_pred_rf = best_rf_model.predict(X_test_pre)
print(classification_report(y_test, y_test_pred_rf, digits=4, zero_division=0))

# SVM
print("\nSVM test results:")
best_svm_model.fit(np.vstack([X_train_pre, X_val_pre]), pd.concat([y_train, y_val]))
y_test_pred_svm = best_svm_model.predict(X_test_pre)
print(classification_report(y_test, y_test_pred_svm, digits=4, zero_division=0))

