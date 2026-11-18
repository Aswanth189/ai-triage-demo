# AI-Assisted Triage Demo

A small, honest machine-learning demo: given a patient's features, predict the
probability of heart disease, as a proxy for how an emergency department might
prioritise patients.

*Built as a learning project, with AI pair-programming support.*

Built to learn and to show the full workflow — data, a baseline model, honest
evaluation, threshold tuning, a model comparison, and an interactive demo —
not to make clinical decisions.

> **Disclaimer — please read this first**
>
> This is a **toy / teaching project, not a real triage tool and not clinical
> validation**. It is trained on a small, old dataset (303 patients, Cleveland,
> 1988), it is not ED triage data, it is not from Singapore, and it has never
> been validated on new patients. **Never use it to make a medical decision.**
> A missed sick patient is the dangerous error here, and this model misses
> about 1 in 5 of them on cross-validation (see Results below).

## The problem

Emergency departments need to prioritise patients quickly. This demo asks a
narrower, learnable version of that question: from 13 routine features —
demographics, symptoms, resting vitals and labs, an exercise stress test and
imaging — can a simple model flag patients who are likely to have heart
disease, so they can be checked sooner?

The answer it gives is a probability, not a diagnosis. A threshold turns that
probability into a flag: above the threshold, "high priority, review urgently";
below it, "lower priority".

## Data

**Source:** UCI Machine Learning Repository — Heart Disease (Cleveland).

- Dataset page: https://archive.ics.uci.edu/dataset/45/heart
- Direct file: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- File in this repo: `heart.csv` (303 rows, 14 columns)

| Column | Meaning |
| --- | --- |
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Resting blood pressure, mm Hg |
| `chol` | Serum cholesterol, mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG: 0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping |
| `ca` | Number of major vessels (0–3) coloured by fluoroscopy |
| `thal` | 3 normal, 6 fixed defect, 7 reversible defect |
| `target` | Diagnosis: 0 = <50% diameter narrowing, 1–4 = >50% narrowing |

Target counts: **0 = 164, 1 = 55, 2 = 36, 3 = 35, 4 = 13** (139 with disease
in total). This project binarises the target — 0 vs 1–4 — because the UCI
codebook formally defines only 0 vs >50% narrowing, and 1–4 have no precise
formal mild/moderate table.

Missing values: `ca` = 4, `thal` = 2, all other columns = 0. The raw file marks
missing as `?`.

**Limitations of the data:** anonymised patients already assessed at the
Cleveland Clinic Foundation, not an ED population, not recent, not Singapore.
Thal is functional perfusion (thallium tracer uptake), ca is anatomical
(vessels with significant blockage seen on imaging) — they can disagree, and
the angiogram / target is the gold standard.

## Method

The model is logistic regression, as a simple, interpretable baseline:

```
median imputation → StandardScaler → LogisticRegression(max_iter=1000)
```

- **Median imputation** fills the 6 missing `ca` / `thal` values with the
  training median (robust to outliers).
- **StandardScaler** rescales every feature to mean 0, standard deviation 1,
  so a feature is not dominant just because its numbers are bigger
  (cholesterol in the hundreds vs sex 0/1).
- **Logistic regression** learns a weight per feature, adds them into one
  score, and squashes it through a sigmoid into a probability between 0 and 1.
  Above 0.5 by default, predict disease; below, predict no disease.

The imputer and scaler are fitted on training data only, inside a scikit-learn
`Pipeline`, so test data never leaks into them.

## Results

### Single split — the headline number, and why not to trust it alone

`train.py` — 80/20 stratified split, `random_state=42`, 61 test patients:

- Accuracy: **0.87** (53/61)
- Confusion matrix:
  ```
  [[27  6]
   [ 2 26]]
  ```
  27 true negatives, 6 false positives, 2 false negatives, 26 true positives.
- Disease precision: **0.81**, disease recall: **0.93**, F1: **0.87**

That 0.93 recall looks great — only 2 sick patients missed — but it rests on
one lucky 61-patient split.

### Cross-validation — the honest estimate

`evaluate_cv.py` — Repeated Stratified 5-fold, 3 repeats (15 folds):

| Metric | Mean ± SD |
| --- | --- |
| Accuracy | 0.843 ± 0.042 |
| Precision | 0.855 ± 0.065 |
| Recall | 0.799 ± 0.078 |
| F1 | 0.822 ± 0.049 |

So the model really catches about **80% of sick patients, not 93%**, and that
number swings a lot (±0.078). **Quote these cross-validated numbers, not the
single-split 0.87.**

- **Recall** = of all actually sick patients, how many were caught.
  `TP / (TP + FN)`. Missing a sick patient is the dangerous error.
- **Precision** = of all patients flagged as sick, how many were actually sick.
  `TP / (TP + FP)`. A false alarm costs an extra check.

### Threshold tuning — trading false alarms for fewer misses

`threshold_test.py` — same 61-patient split, varying the flag threshold:

| Threshold | Confusion matrix | Recall | Precision |
| --- | --- | --- | --- |
| 0.5 | [[27, 6], [2, 26]] | 0.93 | 0.81 |
| 0.4 | [[27, 6], [2, 26]] | 0.93 | 0.81 |
| **0.3** | **[[21, 12], [0, 28]]** | **1.00** | **0.70** |
| 0.2 | [[18, 15], [0, 28]] | 1.00 | 0.65 |

0.4 changes nothing (no test patient had a probability between 0.4 and 0.5).
0.3 catches all 28 sick patients, at the cost of 12 false alarms instead of 6.
0.2 adds false alarms with no extra catches. **0.3 is the best trade-off on
this split, and it is the default in the Streamlit demo** — but that 1.00
recall is a single-split result and still needs cross-validation before it is
quoted as a real number.

### Random Forest comparison

`compare_models.py` — same cross-validation as above, 200 trees:

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.843 ± 0.042 | 0.855 ± 0.065 | 0.799 ± 0.078 | 0.822 ± 0.049 |
| Random Forest | 0.834 ± 0.050 | 0.853 ± 0.065 | 0.775 ± 0.083 | 0.809 ± 0.060 |

Random Forest was slightly worse on every metric, especially recall, and more
variable. With only 303 patients it has no room to use its extra power without
overfitting. Logistic regression was kept — it is simpler, and its weights can
be read and explained.

### What the model learned

Weights from the single-split model (scaled features, 1 unit = 1 SD;
intercept −0.08). Positive pushes towards disease, negative pushes away:

- Strongest positive: `ca` +1.11, `thal` +0.68, `sex` +0.66, `cp` +0.54,
  `exang` +0.38, `slope` +0.35, `trestbps` +0.31, `restecg` +0.22,
  `chol` +0.22, `oldpeak` +0.15
- Strongest negative: `thalach` −0.35, `fbs` −0.22, `age` −0.10

`ca` is by far the strongest single signal, which matches the data: disease
rates rise from 26% at `ca` = 0 to 85% at `ca` = 3.

## Project structure

```
heart.csv            UCI Heart Disease (Cleveland), 303 patients
explore.py           Load the data, print shape, head, target counts, missing values
train.py             Baseline: train/test split, logistic regression, confusion matrix + report
evaluate_cv.py       Repeated stratified 5-fold cross-validation, mean ± SD
threshold_test.py    Try thresholds 0.5 / 0.4 / 0.3 / 0.2, show the recall/precision trade-off
compare_models.py    Logistic regression vs Random Forest, same cross-validation
app.py               Streamlit demo: sliders for a patient, live probability, threshold slider
requirements.txt     pandas, scikit-learn, streamlit
```

## Setup and running

Python 3.12, in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, from the project folder:

```bash
# Explore the data
python3 explore.py

# Train and evaluate on one split
python3 train.py

# Honest cross-validated estimate
python3 evaluate_cv.py

# Threshold trade-off
python3 threshold_test.py

# Logistic regression vs Random Forest
python3 compare_models.py

# Interactive demo (opens at http://localhost:8501)
streamlit run app.py
```

The demo defaults to Patient #1 from the dataset — a worrying-looking
false-positive case (typical angina, deep downsloping ST depression, fixed
thallium defect, but `ca` = 0 and target 0) — so you can see a high probability
for a patient who was in fact labelled healthy. Patient #2 is the opposite
risk: a silent high-risk case (`ca` = 3, but a normal thal scan and no pain at
rest) that a symptoms-only view would miss.

## What this does not prove

- It does not prove the model would work in a real ED, on Singapore patients,
  or on any patient outside this 1988 Cleveland sample.
- It does not prove 87% accuracy — the cross-validated accuracy is ~84%, and
  recall is ~80%.
- It does not prove a 0.3 threshold gives 100% recall in general — that was
  one 61-patient split.
- It is not a diagnosis, a triage decision, or a substitute for clinical
  judgement.

## Next steps

- Validate the 0.3 threshold with cross-validation, not just one split.
- Try more data, or a larger, more recent dataset, before claiming anything
  general.
- Add calibration (do the probabilities mean what they say?) and a proper
  held-out test set that is never used for tuning.
