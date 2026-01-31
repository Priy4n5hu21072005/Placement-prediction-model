import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
Base_Dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_dir=os.path.join(Base_Dir,"Data","Campus_Selection.csv")
df = pd.read_csv(Data_dir)

if 'sl_no' in df.columns:
    df.drop(columns='sl_no',inplace=True)
X=df.drop('status',axis=1)
y=df['status']
le_status=LabelEncoder()
y=le_status.fit_transform(y)
le_status.classes_
 # gender,hsc_s,degree_t,workex,specillisation (Categories)
cat_cols=X.select_dtypes(include='object').columns
cat_cols
le=LabelEncoder()
for col in cat_cols:
    X[col]=le.fit_transform(X[col])
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=23,stratify=y
)
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Logisitic regression
logistic_regression =LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled,y_train)
y_prediction=logistic_regression.predict(X_test_scaled)
print(accuracy_score(y_test,y_prediction))
print(classification_report(y_test,y_prediction))
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=23,
)
rf.fit(X_train,y_train)
y_prediction_rf=rf.predict(X_test)
print(accuracy_score(y_test,y_prediction_rf))
print(classification_report(y_test,y_prediction_rf))
MODEL_DIR=os.path.join(Base_Dir,"models")
os.makedirs(MODEL_DIR,exist_ok=True)
joblib.dump(rf,os.path.join(MODEL_DIR,"placement_model.pkl"))
joblib.dump(scaler,os.path.join(MODEL_DIR,"scaler.pkl"))
joblib.dump(le_status,os.path.join(MODEL_DIR,"label_encoder.pkl"))
print("MODEL process successfully")