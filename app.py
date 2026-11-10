import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    for folder in (BASE_DIR, os.path.join(BASE_DIR, "models")):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"{filename} not found next to app.py or in ./models/")

scaler = load_model("scaler.pkl")
kmeans = load_model("kmeans_model.pkl")
diabetes_model = load_model("best_diabetes_model.pkl")


FEATURES = list(scaler.feature_names_in_)


KMEANS_FEATURES = []

CLUSTER_INFO = {
    0: {"name": "Cluster 0", "description": "TODO: describe this group."},
    1: {"name": "Cluster 1", "description": "TODO: describe this group."},
    2: {"name": "Cluster 2", "description": "TODO: describe this group."},
    3: {"name": "Cluster 3", "description": "TODO: describe this group."},
}


def predict_cluster(row):
    """Return {'id', 'name', 'description'} or None while KMEANS_FEATURES is empty."""
    if not KMEANS_FEATURES:
        return None
    if len(KMEANS_FEATURES) != kmeans.n_features_in_:
        raise ValueError(
            f"KMEANS_FEATURES has {len(KMEANS_FEATURES)} names, "
            f"model expects {kmeans.n_features_in_}"
        )


    vec = np.array([[row.get(col, 0) for col in KMEANS_FEATURES]], dtype=float)
    cid = int(kmeans.predict(vec)[0])
    return {"id": cid, **CLUSTER_INFO.get(cid, {"name": f"Cluster {cid}", "description": ""})}


NUMERIC_RANGES = {"age": (1, 110), "height": (80, 230), "weight": (10, 250), "sleep": (2, 14)}
CHOICES = {"sex": {1, 2}, "active": {0, 1}, "smoking": {0, 1, 2}, "alcohol": {0, 1, 2, 3, 4, 5}}


def parse_answers(data):
    out = {}
    for key, (lo, hi) in NUMERIC_RANGES.items():
        try:
            value = float(data[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{key}' is missing or not a number")
        if not lo <= value <= hi:
            raise ValueError(f"'{key}' must be between {lo} and {hi}")
        out[key] = value
    for key, allowed in CHOICES.items():
        try:
            value = int(data[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{key}' is missing")
        if value not in allowed:
            raise ValueError(f"'{key}' has an invalid value")
        out[key] = value
    return out


def risk_level(p):
    if p < 0.33:
        return "low"
    if p < 0.66:
        return "moderate"
    return "high"


@app.route("/")
def index():
    return render_template("questions.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        a = parse_answers(request.get_json(force=True, silent=True) or {})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    bmi = a["weight"] / (a["height"] / 100) ** 2  # computed server-side, never trusted from the browser

    row = {
        "RIDAGEYR": a["age"],
        "RIAGENDR": a["sex"],          # NHANES coding: 1 = male, 2 = female
        "BMXHT": a["height"],
        "BMXWT": a["weight"],
        "BMXBMI": bmi,
        "SLD010H": a["sleep"],
        "physically_active": a["active"],
        "smoking_status": a["smoking"],  # 0 = never, 1 = former, 2 = current
        "alcohol_use": a["alcohol"],     # drinks on a typical day
    }

    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X)
    proba = diabetes_model.predict_proba(X_scaled)[0]
    p = float(proba[list(diabetes_model.classes_).index(1)])  # class 1 = diabetes

    return jsonify(
        {
            "probability": round(p, 4),
            "level": risk_level(p),
            "bmi": round(bmi, 1),
            "cluster": predict_cluster(row),
        }
    )

if __name__ == "__main__":
    app.run(debug=True)
