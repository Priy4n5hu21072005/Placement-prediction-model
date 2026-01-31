import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===================== PATH SETUP =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "Data", "Campus_Selection.csv")

# ===================== LOAD MODEL & ENCODERS =====================
model = joblib.load(os.path.join(MODEL_DIR, "placement_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH)

# Drop unnecessary column
if 'sl_no' in df.columns:
    df.drop(columns='sl_no', inplace=True)

# Split features & target
X = df.drop('status', axis=1)
y = label_encoder.transform(df['status'])

# Encode categorical features
cat_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# ===================== TRAIN-TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=23,
    stratify=y
)

# ===================== PREDICTIONS =====================
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show(block=False)
plt.pause(2)
plt.close()

# ===================== ROC CURVE =====================
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()

# ===================== PLACEMENT CHANCE (%) =====================
sample = X_test.iloc[[0]]   # MUST be 2D

probability = model.predict_proba(sample)[0][1] * 100
print(f"\nPlacement Chance: {probability:.2f}%")
