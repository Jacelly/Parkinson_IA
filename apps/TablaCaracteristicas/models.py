from django.db import models
from apps.Sujeto.models import Sujeto
from apps.ImagenMascara.models import ImagenMascara
# Create your models here.
class TablaCaracteristicas(models.Model):
    id_tablaC = models.AutoField(primary_key=True)
    nombrePaciente = models.CharField(max_length=30)
    curtosisI = models.DecimalField(max_digits=9,decimal_places=7)
    areaI = models.DecimalField(max_digits=9,decimal_places=7)
    redondezI = models.DecimalField(max_digits=9,decimal_places=7)
    perimetroI = models.DecimalField(max_digits=9,decimal_places=7)
    varianzaI = models.DecimalField(max_digits=9,decimal_places=7)
    desviEstandI = models.DecimalField(max_digits=9,decimal_places=7)
    enlogacionF = models.DecimalField(max_digits=9,decimal_places=7)
    flagnessF = models.DecimalField(max_digits=9,decimal_places=7)
    volumenF = models.DecimalField(max_digits=9,decimal_places=7)
    cantidad = models.IntegerField(null=True, blank=True)
    id_sujeto = models.ForeignKey(Sujeto,null=True,blank=True,on_delete=models.CASCADE)
    id_mask = models.ForeignKey(ImagenMascara,null=True,blank=True,on_delete=models.CASCADE)
    def __str__(self):
        return '{}'.format(self.nombrePaciente)