
from django.db import models

class Prediction(models.Model):
    flower_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class ModelAnalysis(models.Model):
    accuracy = models.FloatField()
    loss = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
