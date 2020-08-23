from django import forms
from apps.Sujeto.models import Sujeto
#from betterforms.multiform import MultiModelForm
class SujetoForm(forms.ModelForm):

	class Meta:
		model = Sujeto

		fields = [
			
			'nombre',
			'apellido',

		]
		labels = {
			
			'nombre': 'Nombre',
			'apellido':'Apellido',
		}
		widgets = {
			
			'Nombre': forms.TextInput(attrs={'class':'form-control'}),
			'Apellido': forms.TextInput(attrs={'class':'form-control'}),

		}