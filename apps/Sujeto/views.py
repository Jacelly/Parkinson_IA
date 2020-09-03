from django.http import HttpResponse
from django.shortcuts import render,redirect
from apps.Sujeto.forms import SujetoForm
from django.contrib import messages
from apps.Doctor.models import Doctor
import time
from django.http import HttpResponseRedirect
from apps.Sujeto.models import Sujeto
# Create your views here.
def index(request):
    return render(request,'Sujeto/index.html')


def SujetoRegister(request):
    print("MIRAAAAA: ",request.method)
    if request.method == 'POST':
        form1 = SujetoForm(request.POST)

        if form1.is_valid():
            form1=form1.save()
            print(request.user.id)
            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()): #request.user.id me devuelve el id del usuario que a iniciado sesion
                messages.success(request, 'Registro del paciente '+ str(request.POST['nombre'])+' '+str(request.POST['apellido'])+ ' ha sido creado con éxito.')
            
                if(Sujeto.objects.filter(nombre=form1.nombre,apellido=form1.apellido).exists() == False):
                    Sujeto.objects.create(nombre=form1.nombre,apellido=form1.apellido)
                    print("listo: ",Sujeto)
                return redirect('home_doctor')
                #return redirect('MriRegister')
                #return HttpResponseRedirect('/imagenMRI/MriRegister')
            messages.success(request, 'Registro ha sido creado con éxito.')
            

            return redirect('home_administrador')
        else:
            messages.warning(request, 'Su registro no se ha podido guardar.')
            if(Doctor.objects.filter(usuario_ptr_id=request.user.id).exists()):
                return redirect('home_doctor')
            return redirect('home_administrador')
    else:
        print("que paso")
        form1 = SujetoForm()

        
    return render(request, 'ImagenMRI/sujetoForm.html', {
        'form1': form1,
    })