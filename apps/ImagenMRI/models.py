from django.db import models
from apps.Sujeto.models import Sujeto
from model_utils import Choices
# Create your models here.
class ImagenMRI(models.Model):
    TIPOS = Choices(
        ('T1', 'T1'),
        ('FLAIR', 'FLAIR'),
    )
    id_mri = models.AutoField(primary_key=True)
    imagen = models.FileField(upload_to='ImagenesMRI/',blank=False, null=False,max_length=1000)
    #nombre = models.CharField(max_length=30)
    tipo = models.CharField(max_length=5,choices=TIPOS)
    #tamanio = models.CharField(max_length=20)
    id_sujeto = models.ForeignKey(Sujeto,null=False,blank=False,on_delete=models.CASCADE)
    descripcion = models.CharField(max_length=150,null=False, blank=False,default="Ninguna")
    def __str__(self):
        return '{}'.format(self.imagen)