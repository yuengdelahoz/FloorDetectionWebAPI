from django.conf.urls import url
from neural_net_api import views

urlpatterns = [
    url(r'^floordetection/$', views.FloorDetection.as_view()),
    url(r'^objectdetection/$', views.ObjectDetection.as_view()),
]
