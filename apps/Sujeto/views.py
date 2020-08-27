from django.http import HttpResponse
from django.shortcuts import render,redirect
from apps.Sujeto.forms import SujetoForm
from django.contrib import messages
from apps.Doctor.models import Doctor
# Create your views here.
def index(request):
    return render(request,'Sujeto/index.html')


def SujetoRegister(request):
    if request.method == 'POST':
        form1 = SujetoForm(request.POST)
        if form1.is_valid():
            form1.save()
            messages.success(request, 'Registro ha sido creado con éxito.')
            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()): #request.user.id me devuelve el id del usuario que a iniciado sesion
                return redirect('home_doctor')
            return redirect('home_administrador')
        else:
            messages.warning(request, 'Su registro no se ha podido guardar.')
            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
                return redirect('home_doctor')
            return redirect('home_administrador')
    else:
        form1 = SujetoForm()
        
    return render(request, 'ImagenMRI/sujetoForm.html', {
        'form1': form1,
    })