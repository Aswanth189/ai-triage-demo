import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv('heart.csv', header=None, names=cols, na_values='?')
df['target'] = (df['target'] > 0).astype(int)
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

# predict_proba gives the actual probabilities, not just 0/1
# [:, 1] takes the probability of class 1, disease
probs = model.predict_proba(X_test)[:, 1]

for t in [0.5, 0.4, 0.3, 0.2]:
    pred = (probs >= t).astype(int)
    print(f"\nThreshold {t}:")
    print(confusion_matrix(y_test, pred))
    print(f"recall: {recall_score(y_test, pred):.2f}  precision: {precision_score(y_test, pred):.2f}")
