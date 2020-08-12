# Generated by Django 3.0.8 on 2020-07-29 14:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('ImagenMRI', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImagenMascara',
            fields=[
                ('id_mask', models.AutoField(primary_key=True, serialize=False)),
                ('nombre', models.CharField(max_length=30)),
                ('tamanio', models.CharField(max_length=20)),
                ('id_mri', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='ImagenMRI.ImagenMRI')),
            ],
        ),
    ]
