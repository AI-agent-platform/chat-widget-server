from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_form, name='chatbot_form'),
    path('api/', views.chatbot_api, name='chatbot_api'),
    path('api/upload-file/', views.chatbot_file_upload, name='chatbot_file_upload'),
    path('api/query/', views.chatbot_rag_query, name='chatbot_rag_query'),
    path('admin/<str:company_name>/<str:uid>/', views.business_owner_agent_api, name='business_owner_agent_api'),
    path('client/<str:company_name>/<str:uid>/', views.client_agent_api, name='client_agent_api'),
    path('download-instructions-html/<str:company_name>/<str:uid>/', views.download_dual_agent_html, name='download_dual_agent_html'),
]