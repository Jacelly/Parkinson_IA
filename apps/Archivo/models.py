from django.db import models

# Create your models here.
class CSV(models.Model):
	id_csv= models.AutoField(primary_key=True)
	documento = models.FileField(upload_to='DATASET_HABLA/',blank=False, null=False)
	titulo=models.CharField(max_length=20,null=True)
	descripcion=models.CharField(max_length=200,null=True)
	class Meta:
		ordering = ['id_csv']
	def __str__(self):
		return self.titulo
