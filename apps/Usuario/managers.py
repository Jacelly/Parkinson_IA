from django.db import models



class AdministradorManager(models.Manager):
	def get_queryset(self):
		return super(AdministradorManager,self).get_queryset().filter(is_staff= True)