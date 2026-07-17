import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import pickle

# Generisanje podataka - SADA VIŠE KVAROVA
np.random.seed(42)
n = 10000  # više uzoraka

air_temp = np.random.normal(300, 20, n)
process_temp = air_temp + np.random.normal(10, 5, n)
rotational_speed = np.random.normal(1500, 300, n)
torque = np.random.normal(40, 10, n)
tool_wear = np.random.uniform(0, 250, n)
product_type = np.random.choice(['L', 'M', 'H'], n, p=[0.5, 0.3, 0.2])

# Pravljenje kvarova - VIŠE KVAROVA
failure = []
for i in range(n):
    # Skor od 0-3
    score = 0
    if tool_wear[i] > 180:
        score += 1
    if torque[i] > 50 and rotational_speed[i] > 1700:
        score += 1
    if process_temp[i] > 330:
        score += 1
    if rotational_speed[i] < 1000 and torque[i] > 45:
        score += 1
    
    # Veća šansa za kvar
    if score >= 3:
        failure.append(np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.2, 0.15, 0.15]))
    elif score >= 2:
        failure.append(np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.25, 0.15, 0.1, 0.1]))
    elif score >= 1:
        failure.append(np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.05, 0.05]))
    else:
        failure.append(0)

df = pd.DataFrame({
    'air_temp': air_temp,
    'process_temp': process_temp,
    'rotational_speed': rotational_speed,
    'torque': torque,
    'tool_wear': tool_wear,
    'product_type': product_type,
    'failure': failure
})

print(f"Raspored kvarova:\n{df['failure'].value_counts().sort_index()}")

# One-hot encoding
df = pd.get_dummies(df, columns=['product_type'], prefix='type')
X = df.drop('failure', axis=1)
y = df['failure']

# Podjela sa stratifikacijom (da zadrži raspored kvarova)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set - raspored: {y_train.value_counts().sort_index().tolist()}")
print(f"Test set - raspored: {y_test.value_counts().sort_index().tolist()}")

# Skaliranje
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic Regression - dodajemo class_weight
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_s, y_train)
print(f"LR accuracy: {lr.score(X_test_s, y_test):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train_s, y_train)
print(f"RF accuracy: {rf.score(X_test_s, y_test):.4f}")

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=3)
xgb_model.fit(X_train_s, y_train)
print(f"XGB accuracy: {xgb_model.score(X_test_s, y_test):.4f}")

# SHAP
explainer = shap.TreeExplainer(rf, X_train_s)

# Čuvanje modela
models = {
    'scaler.pkl': scaler,
    'logistic_regression.pkl': lr,
    'random_forest.pkl': rf,
    'xgboost_model.pkl': xgb_model,
    'shap_explainer.pkl': explainer,
    'feature_names.pkl': X.columns.tolist()
}

for name, obj in models.items():
    pickle.dump(obj, open(name, 'wb'))

print("\n✅ Svi modeli sačuvani!")
print(f"Feature names: {X.columns.tolist()}")