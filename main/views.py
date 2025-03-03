import json

from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os


from .classify import *

def home(request):
    return render(request, 'site/index.html')

def define(request):
    return render(request, 'site/define.html')

def result(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        image_path = os.path.join('main/static/uploads', image.name)

        path = default_storage.save(image_path, ContentFile(image.read()))

        pred, probs, max_prob = classify_image(image_path)

        json_data = {
            "success": "true",
            "prediction": pred,
            "probabilities": probs,
            "confidence": max_prob * 100,
            "error": "None"
        }
        path = path.replace('main', '', 1)

        return render(request, 'site/result.html', {
            'image_url': path,
            'json_data': json_data
        })

    return redirect('define')