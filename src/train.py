# src/train.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():

    data = load_iris()

    X = data.data
    y = data.target

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    return model


if __name__ == "__main__":

    model = train_model()

    joblib.dump(
        model,
        "models/model.pkl"
    )

    print("Model saved")