from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: redirect('/chat/')),  # Redirects root to /chat/
    path('chat/', include('chatbot.urls')),        # Includes app URLs at /chat/
]