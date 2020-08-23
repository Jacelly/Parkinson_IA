from django.contrib import admin
from apps.Archivo.models import CSV
# Register your models here. Se registra la aplicacion para que podamos administrar con la pagina de django
admin.site.register(CSV)
