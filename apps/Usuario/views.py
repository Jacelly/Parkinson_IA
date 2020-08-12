from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import permission_required
from django.shortcuts import render, redirect

@login_required
def home(request):
	user=request.user
	if user.has_perm('Usuario.is_administrador'):
		return redirect('home_administrador')
	elif user.has_perm('Usuario.is_doctor'):
		return redirect('home_doctor')
	else:
		return render(request,template_name='base.html')

@permission_required('Usuario.is_doctor')
def home_doctor(request):
	return render(request,template_name='base/base.html')

@permission_required('Usuario.is_administrador')
def home_administrador(request):
	return render(request,template_name='base/base.html')