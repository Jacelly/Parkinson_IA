from django.db import models
from apps.Usuario.models import Usuario

# Create your models here.
class Doctor(Usuario):
    telefono = models.CharField(max_length=20)
    cedula =  models.CharField(max_length=10,unique=True,null=True)
    class Meta:
        db_table = "Doctor"
