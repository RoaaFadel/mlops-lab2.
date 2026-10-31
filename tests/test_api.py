import joblib

VALID = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_health_contract(client):
    """API contract: /health answers 200 with the agreed JSON shape."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_loads(trained_model):
    """The saved model file can be loaded and can predict."""
    model = joblib.load(trained_model)
    assert hasattr(model, "predict")
    assert model.predict([[5.1, 3.5, 1.4, 0.2]]).shape == (1,)


def test_predict_contract(client):
    """API contract: /predict returns class_id (int) and class_name (str)."""
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"class_id", "class_name"}
    assert isinstance(body["class_id"], int)
    assert body["class_name"] in {"setosa", "versicolor", "virginica"}


def test_input_schema_rejects_bad_input(client):
    """Input schema: missing field, wrong type and negative value give 422."""
    missing = {k: v for k, v in VALID.items() if k != "petal_width"}
    assert client.post("/predict", json=missing).status_code == 422

    wrong_type = {**VALID, "sepal_length": "abc"}
    assert client.post("/predict", json=wrong_type).status_code == 422

    negative = {**VALID, "sepal_length": -5}
    assert client.post("/predict", json=negative).status_code == 422


def test_edge_case_extreme_values(client):
    """Edge case: values at/over the allowed limit are handled safely."""
    just_inside = {k: 19.9 for k in VALID}
    assert client.post("/predict", json=just_inside).status_code == 200

    too_big = {**VALID, "petal_length": 20}
    assert client.post("/predict", json=too_big).status_code == 422


def test_prediction_is_deterministic(client):
    """Same input must always give the same prediction."""
    first = client.post("/predict", json=VALID).json()
    second = client.post("/predict", json=VALID).json()
    assert first == second
    assert first == {"class_id": 0, "class_name": "setosa"}
