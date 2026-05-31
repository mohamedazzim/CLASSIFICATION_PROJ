"""Train a simple pipeline and save the model artifact.
This script auto-detects a CSV in the current working directory (prefers files with 'attrition' in the name),
trains a RandomForest pipeline, and saves it to models/employee_attrition_model.pkl.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as e:
    print('Missing dependency:', e)
    raise

cwd = Path.cwd()
csvs = sorted(cwd.glob('*.csv'))
if not csvs:
    raise FileNotFoundError('No CSV file found in the current working directory.')

data_path = next((p for p in csvs if 'attrition' in p.name.lower()), csvs[0])
print('Using dataset:', data_path.name)

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

TARGET = 'Attrition'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset columns: {df.columns.tolist()}")

# Prepare X/y
y = df[TARGET].astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1})
if y.isna().any():
    raise ValueError('Target contains values other than Yes/No or missing values.')

X = df.drop(columns=[TARGET])

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print('Numeric features:', len(numeric_features), 'Categorical features:', len(cat_features))

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', make_ohe())]), cat_features),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced_subsample'))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Training...')
pipeline.fit(X_train, y_train)

models_dir = cwd / 'models'
models_dir.mkdir(exist_ok=True)
model_path = models_dir / 'employee_attrition_model.pkl'
joblib.dump(pipeline, model_path)
print('Model saved to:', model_path.resolve())

# Optional quick evaluation
try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print('Test accuracy:', round(accuracy_score(y_test, preds), 4))
    print('Test ROC-AUC:', round(roc_auc_score(y_test, probs), 4))
except Exception:
    pass
