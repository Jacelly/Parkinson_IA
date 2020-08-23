from django import forms
from apps.Archivo.models import CSV
class DocumentoForm(forms.ModelForm):

	class Meta:
		model = CSV

		fields = [
			'id_csv',
			'documento',
			'titulo',
			'descripcion',

		]
		labels = {
			'id_csv': 'Id',
			'documento': 'Documento',
			'titulo':'Titulo',
			'descripcion':'Descripción',
		}
		widgets = {
			'id_csv': forms.TextInput(attrs={'class':'form-control'}),
			
			'titulo': forms.TextInput(attrs={'class':'form-control'}),
			'descripcion': forms.TextInput(attrs={'class':'form-control'}),

		}