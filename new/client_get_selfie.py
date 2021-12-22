import http.client
import requests

#conn = http.client.HTTPConnection("localhost", 5000)
url = 'http://localhost:5000/take_selfie'
#conn.request("GET", "/take_selfie")
#response = conn.getresponse()
#print(response.json())
#print(response.status, response.reason)
with requests.Session() as s:
    r = s.get(url)
    print(r.json())