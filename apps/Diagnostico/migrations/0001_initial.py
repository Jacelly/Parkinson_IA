# Generated by Django 3.0.8 on 2020-08-18 20:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('Overlay', '0001_initial'),
        ('Sujeto', '0001_initial'),
        ('ImagenMascara', '0001_initial'),
        ('ImagenMRI', '0001_initial'),
        ('TablaCaracteristicas', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Diagnostico',
            fields=[
                ('id_diag', models.AutoField(primary_key=True, serialize=False)),
                ('id_mask', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ImagenMascara.ImagenMascara')),
                ('id_mri', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ImagenMRI.ImagenMRI')),
                ('id_overlay', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='Overlay.Overlay')),
                ('id_sujeto', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='Sujeto.Sujeto')),
                ('id_tablaC', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='TablaCaracteristicas.TablaCaracteristicas')),
            ],
            options={
                'db_table': 'Diagnostico',
            },
        ),
    ]
