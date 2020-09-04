from django.urls import path
from apps.Diagnostico import views

#Incluyo todas la urls de la apliccacion, para que sean leidas en las urls globales de django
urlpatterns = [
	path('precisionesCsv_Habla/', views.precisionesCsv_Habla, name="precisionesCsv_Habla"),
	path('diagnoticoPorCsv_Habla', views.diagnoticoPorCsv_Habla, name='diagnoticoPorCsv_Habla'),
	path('pruebasMLCsv_Habla/', views.pruebasMLCsv_Habla, name='pruebasMLCsv_Habla'),

	path('pruebasMLCsv_Habla1/', views.DocumentoAddTOtest, name='DocumentoAddTOtest'),


	path('barraCargaModeloDiagPorMRI', views.barraCargaModeloDiagPorMRI, name='barraCargaModeloDiagPorMRI'),
	path('diagnoticoPorMRI', views.diagnoticoPorMRI, name='diagnoticoPorMRI'),

	path('disponible/', views.DiagnosticoDisponible.as_view(), name='diagnostico_disponible'),
	path('disponibleToDoctor/', views.DiagnosticoDisponibleToDoctor.as_view(), name='diagnostico_disponibleToDoctor'),

	path('EditarDiagObser/<int:id_diag>', views.EditarDiagObser, name='EditarDiagObser'),
	path('RegistroDiagEliminar/<int:id_diag>', views.RegistroDiagDelete, name='RegistroDiagDelete'),
	path('RegistroDiagEditar/<int:id_diag>', views.EditarDiagObser, name='RegistroDiagEditar'),
	path('EditarDiagnostico/<pk>',views.EditarDiagnostico.as_view(), name='EditarDiagnostico'),
	path('EliminarDiagnostico/<pk>',views.EliminarDiagnostico.as_view(), name='EliminarDiagnostico'),
	path('filtroPacientes', views.busquedaDiagByPaciente, name='diagnostico_busqueda'),
	path('EliminarDiagnosticoToDoctor/<pk>',views.EliminarDiagnosticoToDoctor.as_view(), name='EliminarDiagnosticoToDoctor'),
	path('EditarDiagnosticoToDoctor/<pk>',views.EditarDiagnosticoToDoctor.as_view(), name='EditarDiagnosticoToDoctor'),

	path('feedback_CargarMRI/',views.feedback_CargarMRI,name="feedback_CargarMRI"),
	path('feedback_CargarCSV/',views.feedback_CargarCSV,name="feedback_CargarCSV"),
	path('feedback_DiagHabla/',views.feedback_DiagHabla,name="feedback_DiagHabla"),
	path('feedback_DiagMri/',views.feedback_DiagMri,name="feedback_DiagMri"),
	path('feedback_ListDiagMri/',views.feedback_ListDiagMri,name="feedback_ListDiagMri"),
	path('feedback_ListDiagMriToDoctor/',views.feedback_ListDiagMriToDoctor,name="feedback_ListDiagMriToDoctor"),
	path('feedback_PrecisionesModelsML/',views.feedback_PrecisionesModelsML,name="feedback_PrecisionesModelsML")
]