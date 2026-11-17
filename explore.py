import pandas as pd

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv('heart.csv', header=None, names=cols, na_values='?')

print(df.shape)
print(df.head())
print(df['target'].value_counts())
print(df.isna().sum())
