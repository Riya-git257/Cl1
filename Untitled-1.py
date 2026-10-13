# ============================================================
# BAYESIAN NETWORK FOR HEART DISEASE DIAGNOSIS
# ============================================================

# Install if required:
# pip install pandas pgmpy

import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


# ============================================================
# 1. LOAD HEART DISEASE DATASET
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

data = pd.read_csv(
    url,
    header=None,
    names=columns,
    na_values="?"
)

print("Original Dataset Shape:", data.shape)

# ============================================================
# 2. REMOVE MISSING VALUES
# ============================================================

data = data.dropna()

print("Dataset Shape After Removing Missing Values:", data.shape)


# ============================================================
# 3. CONVERT TARGET INTO BINARY CLASS
# ============================================================

# Original UCI target:
# 0 = No heart disease
# 1,2,3,4 = Heart disease

data["target"] = data["target"].apply(
    lambda x: 0 if x == 0 else 1
)


# ============================================================
# 4. DISCRETIZE CONTINUOUS VARIABLES
# ============================================================

# Age
data["age"] = pd.cut(
    data["age"],
    bins=[0, 40, 60, 100],
    labels=[0, 1, 2]
)

# Resting blood pressure
data["trestbps"] = pd.cut(
    data["trestbps"],
    bins=[0, 120, 140, 300],
    labels=[0, 1, 2]
)

# Cholesterol
data["chol"] = pd.cut(
    data["chol"],
    bins=[0, 200, 240, 600],
    labels=[0, 1, 2]
)

# Maximum heart rate
data["thalach"] = pd.cut(
    data["thalach"],
    bins=[0, 100, 150, 250],
    labels=[0, 1, 2]
)


# ============================================================
# 5. REMOVE ANY NEW MISSING VALUES
# ============================================================

data = data.dropna()

# Convert everything to integer
data = data.astype(int)

print("\nProcessed Dataset:")
print(data.head())

print("\nFinal Dataset Shape:", data.shape)


# ============================================================
# 6. CONSTRUCT BAYESIAN NETWORK
# ============================================================

model = DiscreteBayesianNetwork([
    ("age", "target"),
    ("sex", "target"),
    ("cp", "target"),
    ("trestbps", "target"),
    ("chol", "target"),
    ("fbs", "target"),
    ("restecg", "target"),
    ("thalach", "target"),
    ("exang", "target")
])


# ============================================================
# 7. TRAIN BAYESIAN NETWORK
# ============================================================

# Do NOT specify MaximumLikelihoodEstimator here.
# This works with the pgmpy version being used.

model.fit(data)

print("\nBayesian Network successfully created!")


# ============================================================
# 8. DISPLAY NETWORK STRUCTURE
# ============================================================

print("\nBayesian Network Structure:")

for edge in model.edges():
    print(edge)


# ============================================================
# 9. CREATE INFERENCE OBJECT
# ============================================================

inference = VariableElimination(model)


# ============================================================
# 10. PATIENT DATA FOR DIAGNOSIS
# ============================================================

# Patient:
# Age = 55 years        -> category 1 (40-60)
# Sex = Male            -> 1
# Chest pain = 2
# Resting BP = 130      -> category 1 (120-140)
# Cholesterol = 230     -> category 1 (200-240)
# Fasting sugar = 0
# Rest ECG = 1
# Max heart rate = 140  -> category 1 (100-150)
# Exercise angina = 0

patient = {
    "age": 1,
    "sex": 1,
    "cp": 2,
    "trestbps": 1,
    "chol": 1,
    "fbs": 0,
    "restecg": 1,
    "thalach": 1,
    "exang": 0
}


# ============================================================
# 11. PERFORM BAYESIAN INFERENCE
# ============================================================

result = inference.query(
    variables=["target"],
    evidence=patient
)


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n----------------------------------------")
print("HEART DISEASE DIAGNOSIS")
print("----------------------------------------")

no_disease_probability = result.values[0] * 100
disease_probability = result.values[1] * 100

print(
    "Probability of No Heart Disease :",
    round(no_disease_probability, 2),
    "%"
)

print(
    "Probability of Heart Disease    :",
    round(disease_probability, 2),
    "%"
)


# ============================================================
# 13. FINAL DIAGNOSIS
# ============================================================

if disease_probability > no_disease_probability:
    print("\nDiagnosis: Heart Disease")
else:
    print("\nDiagnosis: No Heart Disease")

print("----------------------------------------")