from django import forms
from apps.Diagnostico.models import Diagnostico
from apps.Archivo.models import CSV
#from apps.Sujeto.models import Sujeto
#from betterforms.multiform import MultiModelForm
#class SujetoForm(forms.ModelForm):

#	class Meta:
#		model = Sujeto

#		fields = [
			
#			'nombre',
#			'apellido',

#		]
#		labels = {
			
#			'nombre': 'Nombre',
#			'apellido':'Apellido',
#		}
#		widgets = {
			
#			'Nombre': forms.TextInput(attrs={'class':'form-control'}),
#			'Apellido': forms.TextInput(attrs={'class':'form-control'}),

#		}

class DiagnosticoForm(forms.ModelForm):
	
	class Meta:
		model = Diagnostico
		fields = [
			'descripcion',
			'is_parkinson'
		]
		labels = {
			
			'descripcion': 'Descripción',
			'is_parkinson':'Evaluar Diagnostico'
		}
		widgets = {
            'descripcion': forms.Textarea(attrs={'class':'form-control','rows':4, 'cols':40}),
		}

class DiagnosticoCSVForm(forms.ModelForm):
	
	class Meta:
		model = CSV

		fields = [
			'documento',
		]
		labels = {
			'documento': 'Documento',
		}
