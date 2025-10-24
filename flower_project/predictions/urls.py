from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.predict_flower, name="predict_flower"),
    path("analysis/", views.show_analysis, name="show_analysis"),
]
