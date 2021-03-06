# Generated by Django 3.0.8 on 2020-08-25 07:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('TablaCaracteristicas', '0002_remove_tablacaracteristicas_areai'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='cantidad',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='curtosisI',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='desviEstandI',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='enlogacionF',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='flagnessF',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='perimetroI',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='redondezI',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='varianzaI',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
        migrations.AlterField(
            model_name='tablacaracteristicas',
            name='volumenF',
            field=models.DecimalField(decimal_places=4, default=0.0, max_digits=10),
        ),
    ]
