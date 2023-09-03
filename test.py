import requests
import json

url = "https://vluhdpr7jc.execute-api.ca-central-1.amazonaws.com/prod"

payload = {
    "input_s3_uri":"s3://test-vod-v120-source71e471f1-5vcytwlc3m1b/test-videos/20200616_VB_trim.mp4"
    }
headers = {
  'x-api-key': 'fN8H3kB1Xu1ZwCTc2m6kJ9mEgWRUDtNW4yidjiSH',
  'Api-Pass': 'password',
  'Content-Type': 'text/plain'
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

print(response.text)