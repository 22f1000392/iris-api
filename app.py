from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

# Train model at startuppython -m venv venvpython -m venv venv
iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)

class_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return {"message": "Iris Classifier API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }