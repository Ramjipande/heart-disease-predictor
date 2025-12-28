# ===============================
# HEART DISEASE MODEL TRAIN FILE
# ===============================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("data/heart_disease.csv")
print("Dataset loaded successfully")


# -------------------------------
# 2. TARGET COLUMN FIX
# -------------------------------
df['Heart Disease Status'] = df['Heart Disease Status'].map({
    'No': 0,
    'Yes': 1
})


# -------------------------------
# 3. FEATURE LIST (MATCH DATASET)
# -------------------------------
numerical_features = [
    'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
    'Triglyceride Level', 'Fasting Blood Sugar',
    'CRP Level', 'Homocysteine Level', 'Sleep Hours'
]

categorical_features = [
    'Gender', 'Exercise Habits', 'Smoking',
    'Family Heart Disease', 'Diabetes',
    'High Blood Pressure', 'Low HDL Cholesterol',
    'High LDL Cholesterol', 'Alcohol Consumption',
    'Stress Level', 'Sugar Consumption'
]


# -------------------------------
# 4. SPLIT X & Y
# -------------------------------
X = df.drop('Heart Disease Status', axis=1)
y = df['Heart Disease Status']


# -------------------------------
# 5. PREPROCESSING PIPELINE
# -------------------------------

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# -------------------------------
# 6. MODEL PIPELINE
# -------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# -------------------------------
# 7. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 8. TRAIN MODEL
# -------------------------------
model.fit(X_train, y_train)
print("Model training completed")


# -------------------------------
# 9. ACCURACY
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# -------------------------------
# 10. SAVE MODEL
# -------------------------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as heart_model.pkl")
