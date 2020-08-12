from django.db import models
from django.contrib.auth.models import AbstractBaseUser,BaseUserManager,PermissionsMixin
from .managers import AdministradorManager #util para crear proxi model de Administrador
from django.utils.translation import  ugettext as _
# Create your models here.
#Personalizo como se va crear un usuario/superusuario ,con campos de interes como en este caso (username,email,nombre.apellido)
class PersonalizadaBaseUserManager(BaseUserManager):
    use_in_migrations = True
    def create_user(self,email,username,nombre,apellido,password=None):
        if not email:
            raise ValueError('The gives email must be set')
        
        user = self.model(
            username=username,
            email = self.normalize_email(email),
            nombre=nombre,
            apellido=apellido
        )
        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_superuser(self,email,username,nombre,apellido,password):
        user = self.create_user(
            email,
            username,
            nombre,
            apellido,
            password=password,
        )
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user
class Usuario(AbstractBaseUser,PermissionsMixin,models.Model):
    email=models.EmailField(unique=True)
    username = models.CharField(max_length=50,null=False,unique=True)
    nombre=models.CharField(max_length=20,null=True)
    apellido=models.CharField(max_length=20,null=True)
    is_active = models.BooleanField(default = True)
    is_staff = models.BooleanField(default = False)
    USERNAME_FIELD = 'username' #especifico cual va ser el username
    REQUIRED_FIELDS = ['email','nombre','apellido'] # removes email from REQUIRED_FIELDS (los campos requeridos)
    objects = PersonalizadaBaseUserManager()
    def get_full_name(self):
        return '{} {}'.format(self.nombre,self.apellido)
    def get_short_name(self):
        return self.nombre
    def get_usuario(self):
        return self.username
    def __str__(self):
        return '{} {}'.format(self.nombre,self.apellido)
    class Meta: #Creo los permisos para los diferentes roles de usuario
        permissions = (
            ('is_doctor',_('Is_Doctor')),
            ('is_administrador',_('Is_administrador')), 
            ('is_cliente',_('Is_Cliente')),     
        )

#PROXY MODEL -->es como una subconsulta a la tabla, muy util para no estar creando muchas tablas en la BD
class Administrador(Usuario):
    objects = AdministradorManager()
    class Meta:
        proxy = True
    def __str__(self):
        return '{} {}'.format(self.nombre,self.apellido)