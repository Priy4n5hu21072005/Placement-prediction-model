import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
Base_Dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Data_dir=os.path.join(Base_Dir,"Data","Campus_Selection.csv")
df = pd.read_csv(Data_dir)
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df['status'].value_counts()
sns.countplot(x='status',data=df)
sns.countplot(x='workex',hue='status',data=df)
sns.boxplot(x='status',y='mba_p',data=df)
plt.show()

