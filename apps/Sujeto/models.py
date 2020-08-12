from django.db import models

# Create your models here.
class Sujeto(models.Model):
    id_sujeto = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=30)
    apellido = models.CharField(max_length=30)
    isPD = models.BooleanField(default = False)
    def __str__(self):
        return '{} {}'.format(self.nombre,self.apellido)
    