import os
import requests
#import fitz

file_name = 'cni_sse004.png'
#url = 'http://192.168.20.13:81/api'
url = 'http://localhost:5000/extract_info'
with open(file_name, 'rb') as img:
  name_img = os.path.basename(file_name)
  print(name_img)
  print(img)
  files= {'file': (name_img, img,'multipart/form-data') }  # application/pdf
  with requests.Session() as s:
    r = s.post(url, files=files)
    print(r.json())