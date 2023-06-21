from django.contrib import admin

# Register your models here.
from .models import BudgetList
from .models import NativeItem
from .models import TypeDefinition
from .models import ComputationParam

admin.site.register(BudgetList)
admin.site.register(NativeItem)
admin.site.register(TypeDefinition)
admin.site.register(ComputationParam)