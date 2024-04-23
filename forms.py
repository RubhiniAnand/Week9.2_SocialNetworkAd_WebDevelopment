from django import forms
from .models import *


class socForm(forms.ModelForm):
    class Meta():
        model=socModel
        fields=['Age','EstimatedSalary','Gender_Male']
