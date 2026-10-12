import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("odb.csv")

# Clean 4s and 6s: some values have a trailing "+" (e.g. "407+")
for col in ["4s", "6s"]:
    data[col] = data[col].astype(str).str.replace("+", "", regex=False).astype(int)

# Target: high-average batsman
data["high_average"] = (data["Ave"] >= 40).astype(int)

# Runs, NO, Inns and Ave are left out to avoid target leakage
features = ["Mat", "SR", "100", "50", "0", "4s", "6s"]
X = data[features]
y = data["high_average"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
metrics = {
    "accuracy": round(float(accuracy), 4),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
}

joblib.dump(model, "odi_batsman_model.pkl")
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)
data.to_csv("odi_batting_processed.csv", index=False)

print("Training complete:", metrics)