import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
Base_Dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_dir=os.path.join(Base_Dir,"Data","Campus_Selection.csv")
df = pd.read_csv(Data_dir)
df.head()
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
print(X_test_scaled)
print(X_train_scaled)

