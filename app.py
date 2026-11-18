import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Triage Demo", layout="wide")
st.title("AI-Assisted Triage Demo")
st.write("Enter a patient's features and the model gives a heart-disease risk probability. This is a teaching demo on a small 1988 dataset — **not a real clinical tool**.")

@st.cache_resource
def load_model():
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
    model.fit(X, y)
    return model

model = load_model()

# Threshold slider - default 0.3 from our test
threshold = st.sidebar.slider("Flag threshold", 0.1, 0.9, 0.3, 0.05)
st.sidebar.write("Lower threshold = catch more sick patients, more false alarms.")
st.sidebar.write("Cross-validated: accuracy 0.84, recall 0.80, precision 0.86 at 0.5.")

st.header("Patient features")
# Defaults are Patient #1 from our analysis - the worrying-looking false-positive case
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 29, 77, 63)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    cp = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp) + 1
    trestbps = st.slider("Resting BP (mm Hg)", 94, 200, 145)
    chol = st.slider("Cholesterol (mg/dl)", 126, 564, 233)

with col2:
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
    restecg = ["Normal", "ST-T abnormality", "LV hypertrophy"].index(restecg)
    thalach = st.slider("Max heart rate achieved", 71, 202, 150)
    exang = st.selectbox("Exercise-induced angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

with col3:
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 2.3, 0.1)
    slope = st.selectbox("ST slope", ["Upsloping", "Flat", "Downsloping"])
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1
    ca = st.slider("Major vessels (ca)", 0, 3, 0)
    thal = st.selectbox("Thal scan", ["Normal", "Fixed defect", "Reversible defect"])
    thal = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}[thal]

features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
patient = pd.DataFrame([values], columns=features)

prob = model.predict_proba(patient)[0, 1]

st.header("Result")
st.metric("Probability of heart disease", f"{prob:.1%}")
st.progress(prob)

if prob >= threshold:
    st.error(f"⚠️ HIGH PRIORITY — flag for urgent review (prob {prob:.1%} >= threshold {threshold:.2f})")
else:
    st.success(f"Lower priority (prob {prob:.1%} < threshold {threshold:.2f})")

# What pushed the risk up or down
scaler = model.named_steps['scaler']
clf = model.named_steps['clf']
scaled = scaler.transform(patient)[0]
contributions = sorted(
    zip(features, clf.coef_[0] * scaled),
    key=lambda x: abs(x[1]), reverse=True
)
st.subheader("Top drivers for this patient")
for name, contrib in contributions[:3]:
    direction = "pushes risk UP" if contrib > 0 else "pushes risk DOWN"
    st.write(f"- **{name}**: {contrib:+.2f} — {direction}")

st.caption("Model: logistic regression, UCI Heart Disease (Cleveland, 303 patients). Cross-validated accuracy 0.843 ± 0.042, recall 0.799 ± 0.078, precision 0.855 ± 0.065. Random Forest was tested and was slightly worse, so logistic regression was kept.")
