from django.db import models
from apps.ImagenMRI.models import ImagenMRI
from apps.ImagenMascara.models import ImagenMascara
# Create your models here.
class Overlay(models.Model):
    id_overlay = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=30)
    tamanio = models.CharField(max_length=20)
    id_mri = models.ForeignKey(ImagenMRI,null=True,blank=True,on_delete=models.CASCADE)
    id_mask = models.ForeignKey(ImagenMascara,null=True,blank=True,on_delete=models.CASCADE)
    def __str__(self):
        return '{}'.format(self.nombre)
