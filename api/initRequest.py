import base64
import requests
import json

def warmUp():
    with open('./api/static/sample1.png', 'rb') as f1:
        f1_B = base64.b64encode(f1.read())

    f1_B = f1_B.decode('utf-8')

    with open('./api/static/sample2.png', 'rb') as f2:
        f2_B = base64.b64encode(f2.read())

    f2_B = f2_B.decode('utf-8')

    model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]

    headers = {
    'Content-Type': 'application/json'
    }

    for model in model_names:
        data = {
            "model_name": model,
            "img":[
                {
                    "img1": "data:image/png;base64,"+f1_B,
                    "img2": "data:image/png;base64,"+f2_B,
                }
            ]
        }

        payload = json.dumps(data)

        response = requests.request("POST", "http://localhost/verify", headers=headers, data=payload)

    data = { 
        "img": [
            "data:image/png;base64,"+f1_B
        ]
    }

    payload = json.dumps(data)

    response = requests.request("POST", "http://localhost/analyze", headers=headers, data=payload)
    return response

warmUp()