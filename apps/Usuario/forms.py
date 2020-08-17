from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from apps.Usuario.models import Usuario
from apps.Doctor.models import Doctor


class RegistroFormularioDoctor(UserCreationForm):
	class Meta:	
		model = Doctor
		fields =[
				'nombre',
				'apellido',
				'telefono',
				'email',
				'username',
				'cedula',
			]
		labels = {
				'nombre': 'Nombre',
				'apellido':'Apellido',
				'telefono':'Telefono',
				'email':'Correo',
				'username':'Usuario',
				'cedula':'Cedula',
		}
