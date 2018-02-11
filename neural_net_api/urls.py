from django.conf.urls import url
from neural_net_api import views

urlpatterns = [
    url(r'^fallprevention$', views.FallPrevention.as_view())
]
