from django.db import models
from apps.Sujeto.models import Sujeto
from model_utils import Choices
# Create your models here.
class ImagenMRI(models.Model):
    TIPOS = Choices(
        ('T', 'T1'),
        ('F', 'FLAIR'),
    )
    id_mri = models.AutoField(primary_key=True)
    imagen = models.FileField(upload_to='ImagenesMRI/',blank=True, null=True)
    #nombre = models.CharField(max_length=30)
    tipo = models.CharField(max_length=1,choices=TIPOS)
    #tamanio = models.CharField(max_length=20)
    id_sujeto = models.ForeignKey(Sujeto,null=True,blank=True,on_delete=models.CASCADE)
    descripcion = models.CharField(max_length=150,null=False, blank=False,default="Ninguna")
    def __str__(self):
        return '{} {}'.format(self.imagen,self.tipo)