# Generated by Django 3.0.8 on 2020-08-22 02:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Sujeto', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sujeto',
            name='isPD',
        ),
    ]