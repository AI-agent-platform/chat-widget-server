from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_form, name='chatbot_form'),          # /chat/
    path('api/', views.chatbot_api, name='chatbot_api'),        # /chat/api/
]