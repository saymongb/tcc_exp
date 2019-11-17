import requests
import json

data = {  "models": ["NAIVE","SES"],
        "demands": [1,2,5,6,3,2,6,7,9,5,5,6,10,10,11,12,9,9,8,5,8],
        "horizon": 1,
        "part": "Bico Dosador",
        "place": "São Paulo"
}

url = "http://localhost:8090//forecastingMethods/Statistical/"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

while(True):
    r = requests.post(url, data=json.dumps(data), headers=headers)
    print(r)