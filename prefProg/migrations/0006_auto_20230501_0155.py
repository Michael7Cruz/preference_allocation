# Generated by Django 3.2.18 on 2023-04-30 17:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prefProg', '0005_alter_budgetlist_values'),
    ]

    operations = [
        migrations.AddField(
            model_name='budgetlist',
            name='final_allocation',
            field=models.CharField(default='none', max_length=500),
        ),
        migrations.AlterField(
            model_name='budgetlist',
            name='names',
            field=models.CharField(max_length=500),
        ),
        migrations.AlterField(
            model_name='budgetlist',
            name='values',
            field=models.CharField(max_length=500),
        ),
    ]
