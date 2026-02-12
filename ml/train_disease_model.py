import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Generate synthetic dataset (for demo)
np.random.seed(42)
rows = 5000

data = pd.DataFrame({
    "age": np.random.randint(15, 50, rows),
    "height_cm": np.random.randint(140, 180, rows),
    "weight_kg": np.random.randint(40, 100, rows),
    "hemoglobin": np.random.uniform(7, 15, rows),
    "tiredness": np.random.randint(0, 100, rows),
    "weakness": np.random.randint(0, 100, rows),
    "pale_skin": np.random.randint(0, 100, rows),
    "dizziness": np.random.randint(0, 100, rows),
    "breathless": np.random.randint(0, 100, rows),
    "hair_fall": np.random.randint(0, 100, rows),
    "headache": np.random.randint(0, 100, rows),
    "cold_hand": np.random.randint(0, 100, rows),
    "pica": np.random.randint(0, 100, rows),
    "chest_pain": np.random.randint(0, 100, rows),
    "palpitations": np.random.randint(0, 100, rows),
})

# Auto BMI
data["bmi"] = data["weight_kg"] / ((data["height_cm"]/100)**2)

# Label logic (for synthetic training only)
conditions = []
for i in range(rows):
    if data["hemoglobin"][i] < 10:
        conditions.append("Anemia")
    elif data["bmi"][i] > 30:
        conditions.append("PCOD")
    else:
        conditions.append("Safe")

data["label"] = conditions

X = data.drop("label", axis=1)
y = data["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "ml/models/disease_model.pkl")
joblib.dump(le, "ml/models/label_encoder.pkl")

print("✅ Model trained and saved!")
