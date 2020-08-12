from django.db import models
from apps.ImagenMRI.models import ImagenMRI
# Create your models here.
class ImagenMascara(models.Model):
    id_mask = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=30)
    tamanio = models.CharField(max_length=20)
    id_mri = models.ForeignKey(ImagenMRI,null=True,blank=True,on_delete=models.CASCADE)
    #Especifica como se va guaradar el objet en el admin de django
    def __str__(self):
        return '{}'.format(self.nombre)