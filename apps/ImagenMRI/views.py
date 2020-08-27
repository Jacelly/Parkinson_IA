
import gzip
import glob
import os, shutil
from apps.ImagenMRI.models import ImagenMRI
from django.shortcuts import render,redirect
from apps.ImagenMRI.forms import MRIForm
from django.contrib import messages
from apps.Diagnostico.views import isBraimJPG,saveMRINiftytoJPG,changedim
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

pathMedia="media/"
pathTemporal="tmp/"
pathModelBrain="models/brain_not_brain.h5"


'''
Descomprimir .gz to .nii or .dcm
'''
def unzipG(PathSource,PathFolderfinal,Finalname):
    with gzip.open(PathSource, 'rb') as f_in:
        with open(PathFolderfinal+Finalname, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
'''
Return
    1: archivo permitido
    0: Cuando no hay archivo
    -1: archivo sin formato
    -2: archivo con formato incorrecto
'''
def ValidaImg(request):
    if len(request.FILES)==0:
        return 0
    else:
        name = request.FILES['imagen'].name
        arrayname = name.split(".")
        ultimaPos=len(arrayname)-1
        #la imagen no tiene extension
        if len(arrayname)==0:
            return -1
        elif arrayname[ultimaPos]=="nii" or arrayname[ultimaPos]=="dcm":
            return 1
        elif  arrayname[ultimaPos]=="gz":
            if arrayname[ultimaPos-1]=="nii" or arrayname[ultimaPos-1]=="dcm":
                return 2
            else:
                return -2
        else:
            return -2
'''
Delete All tmp folder
'''
def removeAll():
    for filename in os.listdir(pathMedia+pathTemporal):
        file_path = os.path.join(pathMedia+pathTemporal, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def MriRegister(request):
    if request.method == 'POST':
        form2 = MRIForm(request.POST or None,request.FILES or None)
      
        case = ValidaImg(request)
        #Satisfactorio imagen nii o dcm o gz con los anteriores formatos
        if case==1 or case==2:
            #print(form2)
            if form2.is_valid():
                #form2.save()
                #filen=default_storage.save('media/tmp/'+request.FILES['imagen'].name, request.FILES['imagen'].data)
                #print(type(request.FILES['imagen']))
                pathMRI=pathMedia+pathTemporal+request.FILES['imagen'].name
                pathJPG=""
                data = request.FILES['imagen'] # or self.files['image'] in your form
                with open(pathMRI, 'wb+') as destination:
                    for chunk in data.chunks():
                        destination.write(chunk)
                #path = default_storage.save(pathJPGMRI, ContentFile(data.read()))
                #tmp_file = os.path.join(settings.MEDIA_ROOT, path)

                if case==2:
                    #se guardo el .nii file
                    pathMRI=pathMedia+pathTemporal+request.FILES['imagen'].name.split(".gz")[0]
                    unzipG(pathMedia+pathTemporal+request.FILES['imagen'].name,pathMedia+pathTemporal,request.FILES['imagen'].name.split(".gz")[0])
                    formato = pathMRI.split(".gz")[0].split(".")[(len(pathMRI.split(".gz")[0].split("."))-1)]
                    if formato=="dcm":
                        dcm2nii(pathMRI,pathMRI.split(".dcm")[0]+".nii")
                        pathJPG=request.FILES['imagen'].name.split(".gz")[0].split(".dcm")[0]
                    else:
                        pathJPG=request.FILES['imagen'].name.split(".gz")[0].split(".nii")[0]
                elif case==1:
                    formato = pathMRI.split(".")[(len(pathMRI.split("."))-1)]
                    if formato=="dcm":
                        pathJPG=request.FILES['imagen'].name.split(".dcm")[0]
                    else:
                        pathJPG=request.FILES['imagen'].name.split(".nii")[0]
                casetmp=saveMRINiftytoJPG(pathMRI,pathMedia+pathTemporal,pathJPG)
                print("hola------------------------------------------------------->")
                print(casetmp)
                listaImg= glob.glob(pathMedia+pathTemporal+"*.jpg")
                print(listaImg)
                changedim(listaImg[len(listaImg)-1])
                bandera = isBraimJPG(listaImg[len(listaImg)-1],pathMedia+pathModelBrain)
                if bandera:
                    print("Si entra y es cerebro")
                    form2.save()        
                    messages.success(request, 'Registro ha sido creado con éxito.')
                    removeAll()
                    return redirect('home_administrador')
                else:
                    print("Si entra pero no es cerebro")
                    messages.warning(request, 'Su registro no se ha podido guardar.')
                    removeAll()
                    return redirect('home_administrador')
                #print("pasa")
                #messages.success(request, 'Registro ha sido creado con éxito.')
                #return redirect('home_administrador')
            else:
                print("No entra")
                messages.warning(request, 'Su registro no se ha podido guardar.')
                removeAll()
                return redirect('home_administrador')
        elif case==0:
            #print("Por favor ingresar archivos NIfTI o DICOM")
            messages.warning(request,"Por favor ingresar archivos NIfTI o DICOM")
            return redirect('home_administrador')
            #messages.warning(request, 'Por favor ingresar archivos NIfTI o DICOM')
        elif case==-1:
            #print("Este archivo no tiene extensión, por favor ingresar archivos NIfTI o DICOM")
            messages.warning(request,"Este archivo no tiene extensión, por favor ingresar archivos NIfTI o DICOM")
            return redirect('home_administrador')
        elif case==-2:
            #print("Este archivo no tiene una extensión valida, por favor ingresar archivos NIfTI o DICOM")
            messages.warning(request,"Este archivo no tiene una extensión valida, por favor ingresar archivos NIfTI o DICOM")
            return redirect('home_administrador')
    else:
        form2 = MRIForm()
        
    return render(request, 'ImagenMRI/MriForm.html', {
        'form2': form2,
       
    })