from django.db import models
from apps.ImagenMRI.models import ImagenMRI
from apps.ImagenMascara.models import ImagenMascara
from apps.Overlay.models import Overlay
from apps.Sujeto.models import Sujeto
from apps.TablaCaracteristicas.models import TablaCaracteristicas

# Create your models here.
class Diagnostico(models.Model):
    id_diag = models.AutoField(primary_key=True)
    id_mri = models.ForeignKey(ImagenMRI,null=True,blank=True,on_delete=models.CASCADE)
    id_mask = models.ForeignKey(ImagenMascara,null=True,blank=True,on_delete=models.CASCADE)
    id_overlay = models.ForeignKey(Overlay,null=True,blank=True,on_delete=models.CASCADE)
    id_sujeto = models.ForeignKey(Sujeto,null=True,blank=True,on_delete=models.CASCADE)
    id_tablaC = models.ForeignKey(TablaCaracteristicas,null=True,blank=True,on_delete=models.CASCADE)
    descripcion = models.CharField(max_length=150,null=False, blank=False,default="Ninguna")
    class Meta:
        db_table = "Diagnostico"