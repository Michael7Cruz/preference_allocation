# Generated by Django 4.2.1 on 2023-06-06 20:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prefProg', '0017_alter_nativeitem_item_subtype'),
    ]

    operations = [
        migrations.CreateModel(
            name='ComputationParam',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('types', models.CharField(max_length=1024)),
                ('items', models.CharField(max_length=2048)),
            ],
        ),
    ]
