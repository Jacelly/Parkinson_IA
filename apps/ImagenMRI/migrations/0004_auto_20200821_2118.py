# Generated by Django 3.0.8 on 2020-08-22 02:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ImagenMRI', '0003_auto_20200818_1603'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imagenmri',
            name='nombre',
        ),
        migrations.RemoveField(
            model_name='imagenmri',
            name='tamanio',
        ),
    ]
