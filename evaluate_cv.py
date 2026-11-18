import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv('heart.csv', header=None, names=cols, na_values='?')
df['target'] = (df['target'] > 0).astype(int)
X = df.drop(columns='target')
y = df['target']

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_validate(model, X, y, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1'])

for name in ['accuracy', 'precision', 'recall', 'f1']:
    vals = scores[f'test_{name}']
    print(f"{name}: {vals.mean():.3f} +/- {vals.std():.3f}")
