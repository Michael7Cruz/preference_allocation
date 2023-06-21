from django import forms
from .models import BudgetList

class BudgetingForm(forms.ModelForm):
    class Meta:
        model = BudgetList
        fields = ['name', 'names', 'values', 'income']

