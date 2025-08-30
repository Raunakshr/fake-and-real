"""API views for prediction endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

import joblib

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pipeline.joblib"
LABELS = {0: "fake", 1: "real"}
PIPELINE = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@csrf_exempt
def predict(request):
    """Return a prediction for supplied JSON text payload."""

    if request.method != "POST":
        return HttpResponseBadRequest("POST request required")

    if PIPELINE is None:
        return JsonResponse({"error": "Model not available"}, status=500)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    text = payload.get("text")
    if not text:
        return HttpResponseBadRequest("'text' field required")

    proba = PIPELINE.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = LABELS.get(idx, str(idx))
    return JsonResponse({"label": label, "proba": float(proba[idx])})
