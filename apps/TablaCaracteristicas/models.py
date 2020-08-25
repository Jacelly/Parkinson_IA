from django.db import models
from apps.Sujeto.models import Sujeto
from apps.ImagenMascara.models import ImagenMascara
# Create your models here.
class TablaCaracteristicas(models.Model):
    id_tablaC = models.AutoField(primary_key=True)
    nombrePaciente = models.CharField(max_length=30)
    curtosisI = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    redondezI = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    perimetroI = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    varianzaI = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    desviEstandI = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    volumenF = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    enlogacionF = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    flagnessF = models.DecimalField(max_digits=10,decimal_places=4, default=0.0000)
    cantidad = models.IntegerField(null=False, blank=False)
    id_sujeto = models.ForeignKey(Sujeto,null=True,blank=True,on_delete=models.CASCADE)
    id_mask = models.ForeignKey(ImagenMascara,null=True,blank=True,on_delete=models.CASCADE)
    def __str__(self):
        return '{} {}'.format(self.id_tablaC,self.nombrePaciente)