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

            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
               #messages.success(request, 'Su documento ha sido creado con éxito.')
               return redirect('home_doctor')
            #messages.success(request, 'Su documento ha sido creado con éxito.')
            return redirect('home_administrador')
        else:
            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
               #messages.success(request, 'Su documento no se ha podido guardar.')
               return redirect('home_doctor')
            #messages.error(request, 'Su documento no se ha podido guardar.')
            return redirect('home_administrador')
    else:
        form = DocumentoForm()

    if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
        messages.success(request, 'Su documento ha sido creado con éxito.')
        return render(request, 'archivo/formAddCsv.html', {
            'form': form,
        })

    messages.success(request, 'Su documento ha sido creado con éxito.')
    return render(request, 'archivo/formAddCsv.html', {
        'form': form,
    })