# Generated by Django 3.2.18 on 2023-04-30 16:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prefProg', '0003_auto_20230430_2359'),
    ]

    operations = [
        migrations.AlterField(
            model_name='budgetlist',
            name='income',
            field=models.FloatField(max_length=20),
        ),
        migrations.AlterField(
            model_name='budgetlist',
            name='name',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='budgetlist',
            name='names',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='budgetlist',
            name='values',
            field=models.FloatField(max_length=200),
        ),
    ]
