from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import permission_required
from django.shortcuts import render, redirect
from apps.Doctor.models import Doctor
from django.contrib import messages
from django.contrib.auth.models import Permission
from apps.Usuario.forms import RegistroFormularioDoctor
# Create your views here.
def DoctorCreate(request):
	#administrador = Administrador.objects.get(id=request.user.id)
	if request.method == 'POST':
		form = RegistroFormularioDoctor(request.POST or None ,request.FILES or None)
		if form.is_valid():
			model = Doctor
			template_name = "Usuario/registrarDoctor.html"
			form_class = RegistroFormularioDoctor
			form.save()
			messages.success(request, 'Se ha registrado Doctor con éxito.')
			permisoDoctor = Permission.objects.get(codename='is_doctor')
			my_user=Doctor.objects.get(id=form.save().id)
			my_user.user_permissions.add(permisoDoctor)
			return redirect('home_doctor')
		else:
			messages.error(request, 'No se ha podido registar Doctor.')
			return redirect('home_doctor')
	else:
		form = RegistroFormularioDoctor()
	return render(request, 'Usuario/registrarDoctor.html', {
		'form': form,
		#'admin': administrador
})
@login_required
def home(request):
	user=request.user
	if user.has_perm('Usuario.is_administrador'):
		return redirect('home_administrador')
	elif user.has_perm('Usuario.is_doctor'):
		return redirect('home_doctor')
	else:
		return render(request,template_name='base.html')

@permission_required('Usuario.is_doctor')
def home_doctor(request):
	return render(request,template_name='base/base.html')

@permission_required('Usuario.is_administrador')
def home_administrador(request):
	return render(request,template_name='base/base_admin.html')