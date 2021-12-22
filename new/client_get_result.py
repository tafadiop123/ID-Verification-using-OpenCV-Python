import http.client
import requests

#conn = http.client.HTTPConnection("localhost", 5000)
#conn.request("GET", "/id_matching")
#response = conn.getresponse()
#print(response.status, response.reason)


url = 'http://localhost:5000/id_matching'
with requests.Session() as s:
    r = s.get(url)
    print(r.json())