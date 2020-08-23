from django.urls import path
from apps.Diagnostico import views

#Incluyo todas la urls de la apliccacion, para que sean leidas en las urls globales de django
urlpatterns = [
	path('precisionesCsv_Habla/', views.precisionesCsv_Habla, name="precisionesCsv_Habla"),
	path('diagnoticoPorCsv_Habla', views.diagnoticoPorCsv_Habla, name='diagnoticoPorCsv_Habla'),
	path('pruebasMLCsv_Habla/', views.pruebasMLCsv_Habla, name='pruebasMLCsv_Habla'),
	path('barraCargaModeloDiagPorMRI', views.barraCargaModeloDiagPorMRI, name='barraCargaModeloDiagPorMRI'),
	path('diagnoticoPorMRI', views.diagnoticoPorMRI, name='diagnoticoPorMRI'),
]