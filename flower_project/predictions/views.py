from django.shortcuts import render
from django.http import JsonResponse
import joblib
import numpy as np
from .models import Prediction, ModelAnalysis

# Load your trained model
model = joblib.load("model.pkl")  # save model after training in training.py

def predict_flower(request):
    # Example input (replace with actual request handling)
    features = [5.1, 3.5, 1.4, 0.2]  
    prediction = model.predict([features])[0]
    confidence = np.max(model.predict_proba([features]))

    # Store in database
    Prediction.objects.create(flower_name=prediction, confidence=confidence)

    return JsonResponse({"flower": prediction, "confidence": confidence})

def show_analysis(request):
    analysis = ModelAnalysis.objects.last()
    return JsonResponse({
        "accuracy": analysis.accuracy,
        "loss": analysis.loss
    })
