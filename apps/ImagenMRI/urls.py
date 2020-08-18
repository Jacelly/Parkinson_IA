from django.urls import path
from apps.ImagenMRI import views

#Incluyo todas la urls de la apliccacion, para que sean leidas en las urls globales de django
urlpatterns = [
	path('registrarImagenMRI/', views.ImagenMRICreate, name="registrar_ImagenMRI"),

]