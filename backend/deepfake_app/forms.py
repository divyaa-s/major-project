from django import forms
from .models import DeepfakeImage

class DeepfakeImageForm(forms.ModelForm):
    class Meta:
        model = DeepfakeImage
        fields = ['image']
