from django import forms
from apps.ImagenMRI.models import ImagenMRI
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

class MRIForm(forms.ModelForm):
	
	class Meta:
		model = ImagenMRI
		CHOICES = [('1', 'First'), ('2', 'Second')]
		#exclude = ('primary',)
		fields = [
			
			'imagen',
			'id_sujeto',
			'tipo',
			'descripcion',

		]
		labels = {
			
			'imagen': 'Imagen',
			'id_sujeto':'Cliente',
			'tipo':'Tipo',
			'descripcion':'Descripción',
		}
		widgets = {
			
			'tipo': forms.Select(attrs={'class':'col-sm-12','style':'height:30px;'}),
			'id_sujeto': forms.Select(attrs={'class':'col-sm-12','style':'height:30px;'}),
			'descripcion': forms.Textarea(attrs={'class':'form-control','rows':4, 'cols':40}),

		}
		