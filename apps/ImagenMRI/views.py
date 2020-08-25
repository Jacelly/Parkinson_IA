from django.shortcuts import render,redirect
from apps.ImagenMRI.forms import MRIForm
from django.contrib import messages
#from apps.Diagnostico.views import generateMaskTwoArgument,getMascaraIntensidad
# Create your views here.
#def SujetoRegister(request):
#   if request.method == 'POST':
#        form1 = SujetoForm(request.POST)
      
#        if form1.is_valid():
#            form1.save()
          
#            messages.success(request, 'Registro ha sido creado con éxito.')
#            return redirect('home_administrador')
#        else:
#            messages.warning(request, 'Su registro no se ha podido guardar.')
#            return redirect('home_administrador')
#    else:
#        form1 = SujetoForm()
        
#    return render(request, 'ImagenMRI/sujetoForm.html', {
#        'form1': form1,
       
#    })
def MriRegister(request):
    if request.method == 'POST':
        form2 = MRIForm(request.POST or None,request.FILES or None)
      
        if form2.is_valid():
            form2.save()
          
            messages.success(request, 'Registro ha sido creado con éxito.')
            return redirect('home_administrador')
        else:
            messages.warning(request, 'Su registro no se ha podido guardar.')
            return redirect('home_administrador')
    else:
        form2 = MRIForm()
        
    return render(request, 'ImagenMRI/MriForm.html', {
        'form2': form2,
       
    })