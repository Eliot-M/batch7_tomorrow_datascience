import requests
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/demo_imgs/2.jpg')

url = "http://127.0.0.1:3000/predict"

files = [
    (
        "image",
        open(filename,
            "rb",
        ),
    )
]

response = requests.post(url, files=files)

print(response.text.encode("utf8"))
