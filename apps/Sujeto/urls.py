from django.urls import path
from apps.Sujeto import views

#Incluyo todas la urls de la apliccacion, para que sean leidas en las urls globales de django
urlpatterns = [
	path('ok', views.index, name="Sujeto_index"),
	path('registrarCliente/', views.SujetoRegister, name="registrarCliente"),
]