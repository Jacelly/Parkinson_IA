# Generated by Django 3.0.8 on 2020-08-18 20:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Overlay', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='overlay',
            name='imagen',
            field=models.ImageField(blank=True, null=True, upload_to='ImagenesOverlay/'),
        ),
    ]