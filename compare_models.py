import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv('heart.csv', header=None, names=cols, na_values='?')
df['target'] = (df['target'] > 0).astype(int)
X = df.drop(columns='target')
y = df['target']

logreg = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# No scaler needed for trees - they split on values, they don't care about scale
rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

for name, model in [('Logistic Regression', logreg), ('Random Forest', rf)]:
    scores = cross_validate(model, X, y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1'])
    print(f"\n{name}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        vals = scores[f'test_{metric}']
        print(f"  {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")
