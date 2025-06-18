from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_form, name='chatbot_form'),
    path('api/', views.chatbot_api, name='chatbot_api'),
    path('api/upload-file/', views.chatbot_file_upload, name='chatbot_file_upload'),
    path('api/query/', views.chatbot_rag_query, name='chatbot_rag_query'),  
]