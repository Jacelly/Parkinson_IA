"""proyectoParkinson URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
#DEPENDENCIAS PARA LOGIN
from django.contrib.auth.views import logout_then_login,LoginView
from django.contrib.auth import views as auth_views
from apps.Usuario import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('sujeto/', include('apps.Sujeto.urls'), name='principalPage'), #incluyendo las urls de una aplicacion determinada
    path('overlay/', include('apps.Overlay.urls'), name='overlay'),
    path('imagenMRI/', include('apps.ImagenMRI.urls'), name='imagenMRI'),
    path('imagenMascara/', include('apps.ImagenMascara.urls'), name='imagenMascara'),
    path('tablaFeatures/', include('apps.TablaCaracteristicas.urls'), name='tablaFeatures'),
    path('user/', include('apps.Usuario.urls'), name='user'),
    path('doctor/', include('apps.Doctor.urls'), name='doctor'),
    path('Archivo/',include('apps.Archivo.urls'), name='archivo'),
    path('Diagnostico/',include('apps.Diagnostico.urls'), name='diagnostico'),

    path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', logout_then_login, name="logout"),
    path('', views.home, name='home'),
    path('home_administrador/', views.home_administrador, name="home_administrador"),
    path('home_doctor/', views.home_doctor, name="home_doctor"),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
