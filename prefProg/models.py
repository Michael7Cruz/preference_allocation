from django.db import models
from django.db.models.deletion import CASCADE
# Create your models here.
class BudgetList(models.Model):
    name = models.CharField(max_length=20)
    names = models.CharField(max_length=500)
    values = models.CharField(max_length=500)
    income = models.FloatField(max_length=20)
    def __str__(self):
        return self.name
    
class TypeDefinition(models.Model):
    name = models.CharField(max_length=20)
    def __str__(self):
        return self.name
    
class NativeItem(models.Model):
    name = models.CharField(max_length=32)
    item_type = models.ForeignKey(TypeDefinition, on_delete=models.CASCADE)
    item_subtype = models.CharField(max_length=32, blank=True, default='')
    unit = models.CharField(max_length=12, blank=True, default='')
    price = models.FloatField(max_length=20)
    def __str__(self):
        return str(self.name)
    
class ComputationParam(models.Model):
    hierarchy_field = models.CharField(max_length=2048, blank=True, default='')
    
