from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import pandas as pd
import shutil
from PIL import Image
import os
import cv2
import matplotlib
import base64
# Create your views here.
from .models import FilesUpload


def index(request):
    response = render(request, 'main/index.html')
    print(request.method)
    if(request.method == "POST"):
        print(request.POST)
        print(len((request.POST)))
        if(len(request.POST) == 3):
            return response
        im = request.FILES["photo"]
        os.remove("./media/one.jpeg")
        im.name = "one.jpeg"
        document = FilesUpload.objects.create(file=im)
        document.save()

        print(im.size)
        model = load_model("D:\SEM6\hci\main\model_adv.h5")
        img = image.load_img(
            "./media/one.jpeg", target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = model.predict_classes(img)
        print(p)
        x = p[0, 0]

        if(x == 0):
            test = "POSITIVE"
            z = 1
        else:
            test = "NEGATIVE"
            z = 0
        context = {
            "rep": z,
            "test": test
        }
        response = render(request, 'main/result.html', context)
        return response

    return response


def about(request):
    response = render(request, 'main/about.html')
    return response


def methodology(request):
    response = render(request, 'main/methodology.html')
    return response
