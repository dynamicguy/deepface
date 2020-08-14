import base64
import requests
import json
import random

def warmUp():

    
    
    headers= {}
    
    files = [
        ('image1', open('./api/static/sample1.png','rb')),
        ('image2', open('./api/static/sample2.png','rb'))
    ]

    model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]

    for model in model_names:

        payload = {'model_name': model,
        'distance_metric': 'cosine'}

        response = requests.request("POST", "http://localhost/verify", data=payload, files=files, headers=headers)

    payload = {}
    files2 = [
        ('image', open('./api/static/sample1.png','rb'))
    ]
    headers= {}

    response = requests.request("POST", "http://localhost/analyze", headers=headers, data = payload, files = files2)
    return response

warmUp()