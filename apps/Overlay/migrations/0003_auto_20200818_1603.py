# Generated by Django 3.0.8 on 2020-08-18 21:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ImagenMascara', '0003_imagenmascara_descripcion'),
        ('Overlay', '0002_overlay_imagen'),
    ]

    operations = [
        migrations.AlterField(
            model_name='overlay',
            name='id_mask',
            field=models.ForeignKey(blank=True, default='Ninguna', null=True, on_delete=django.db.models.deletion.CASCADE, to='ImagenMascara.ImagenMascara'),
        ),
    ]