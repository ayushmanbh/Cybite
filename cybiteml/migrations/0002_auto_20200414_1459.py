# Generated by Django 3.0.5 on 2020-04-14 09:29

import cybiteml.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cybiteml', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='file',
            field=models.FileField(blank=True, upload_to=cybiteml.models.user_directory_path),
        ),
    ]
