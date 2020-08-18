from django.db import models
from apps.Sujeto.models import Sujeto
# Create your models here.
class ImagenMRI(models.Model):
    id_mri = models.AutoField(primary_key=True)
    imagen = models.ImageField(upload_to='ImagenesMRI/',blank=True, null=True)
    nombre = models.CharField(max_length=30)
    tipo = models.CharField(max_length=5)
    tamanio = models.CharField(max_length=20)
    id_sujeto = models.ForeignKey(Sujeto,null=True,blank=True,on_delete=models.CASCADE)
    descripcion = models.CharField(max_length=150,null=False, blank=False,default="Ninguna")
    def __str__(self):
        return '{} {}'.format(self.nombre,self.tipo)