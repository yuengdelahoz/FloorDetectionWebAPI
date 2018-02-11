from django.conf.urls import url
from neural_net_api import views

urlpatterns = [
    url(r'^objectdetection$', views.ObjectDetection.as_view()),
    url(r'^waterdetection$', views.WaterDetection.as_view()),
    url(r'^floordetection$', views.FloorDetection.as_view())
]
