from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from apps.Archivo.forms import DocumentoForm
from apps.Archivo.models import CSV
from django.contrib import messages
# Create your views here.

def DocumentoAdd(request):
	if request.method == 'POST':
		form = DocumentoForm(request.POST or None ,request.FILES or None)
		if form.is_valid():
			form.save()
			messages.success(request, 'Su documento ha sido creado con éxito.')
			#instanciaCsv=CSV.objects.last()
			#path="media/"+str(instanciaCsv.documento)
			#print(path)
			
			return redirect('home_administrador')
		else:
			messages.error(request, 'Su documento no se ha podido guardar.')
			return redirect('home_administrador')
	else:
		form = DocumentoForm()
	return render(request, 'archivo/formAddCsv.html', {
		'form': form,
	})