from django.shortcuts import render
from apps.Usuario.models import Administrador
from apps.ImagenMRI.models import ImagenMRI
from apps.ImagenMRI.forms import RegistroFormularioImagenMRI

# Create your views here.
def ImagenMRICreate(request):
	administrador = Administrador.objects.get(id=request.user.id)
	if request.method == 'POST':
		form = RegistroFormularioImagenMRI(request.POST or None ,request.FILES or None)
		if form.is_valid():
			model = ImagenMRI
			template_name = "ImagenMRI/registrarImagenMRI.html"
			form_class = RegistroFormularioImagenMRI
			form.save()
			messages.success(request, 'Se ha registrado la imagen de resonacia magnética con éxito.')
			#permisoDoctor = Permission.objects.get(codename='is_doctor')
			#my_user=Doctor.objects.get(id=form.save().id)
			#my_user.user_permissions.add(permisoDoctor)
			return redirect('home_admin')
		else:
			messages.error(request, 'No se ha podido registar Doctor.')
			return redirect('home_admin')
	else:
		form = RegistroFormularioImagenMRI()
	return render(request, 'ImagenMRI/registrarImagenMRI.html', {
		'form': form,
		#'admin': administrador
})