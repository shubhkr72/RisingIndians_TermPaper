from __future__ import annotations
import os

import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor import ModelPredictor

app = Flask(__name__)
CORS(app)  # allow all origins
predictor = ModelPredictor("models")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "available_models": predictor.available_models(),
        }
    )


@app.post("/api/v1/predict")
def predict_endpoint():
    started_at = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    title = (payload.get("title") or "").strip()
    model_name = (payload.get("model") or "hybrid").strip()

    if not text:
        return jsonify({"status": "error", "message": "Request JSON must include 'text'."}), 400

    try:
        prediction = predictor.predict(text=text, title=title, model_name=model_name)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    return jsonify(
        {
            "status": "success",
            "prediction": prediction,
            "processing_time_ms": elapsed_ms,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
