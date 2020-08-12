from django.contrib import admin
from apps.Usuario.models import Usuario,Administrador
from django.contrib.auth.admin import  UserAdmin

# Register your models here.
class PersonalizadoUserAdmin(UserAdmin):
	fieldsets = ()
	add_fieldsets = (
		(None,{
			'fields':('nombre','apellido','email','password1','password2'),
		}),
	)
	list_display = ('email','is_active',"is_staff",)
	serach_fields = ('email',)	
	ordering = ('email',)
	
admin.site.register(Usuario,PersonalizadoUserAdmin)
admin.site.register(Administrador)