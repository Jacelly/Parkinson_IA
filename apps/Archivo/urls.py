from django.urls import path
from apps.Archivo import views

#Incluyo todas la urls de la apliccacion, para que sean leidas en las urls globales de django
urlpatterns = [
	#path('DocuDisponible/', views.DocumentoLista.as_view(), name='documentos_disponible'),
	path('documentoAdd/', views.DocumentoAdd, name="documento_agregar"),
]