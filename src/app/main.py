"""FastAPI service exposing the trained model."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        Path(__file__).resolve().parents[2] / "models" / "model.joblib",
    )
)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="AI381 Lab 2 - Iris Classifier", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="cm")
    sepal_width: float = Field(..., gt=0, lt=20, description="cm")
    petal_length: float = Field(..., gt=0, lt=20, description="cm")
    petal_width: float = Field(..., gt=0, lt=20, description="cm")


class Prediction(BaseModel):
    class_id: int
    class_name: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    class_id = int(state["model"].predict(row)[0])
    return Prediction(class_id=class_id, class_name=CLASS_NAMES[class_id])
