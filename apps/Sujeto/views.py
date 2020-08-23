from django.http import HttpResponse
from django.shortcuts import render,redirect
from apps.Sujeto.forms import SujetoForm
from django.contrib import messages
# Create your views here.
def index(request):
    return render(request,'Sujeto/index.html')


def SujetoRegister(request):
    if request.method == 'POST':
        form1 = SujetoForm(request.POST)
      
        if form1.is_valid():
            form1.save()
          
            messages.success(request, 'Registro ha sido creado con éxito.')
            return redirect('home_administrador')
        else:
            messages.warning(request, 'Su registro no se ha podido guardar.')
            return redirect('home_administrador')
    else:
        form1 = SujetoForm()
        
    return render(request, 'ImagenMRI/sujetoForm.html', {
        'form1': form1,
       
    })