import boto3
import io

s3 = boto3.resource('s3')

bucket_name = 'sm-ball-tracking-output-labels'
key = '20200616_VB_trim.mp4'

bucket = s3.Bucket(bucket_name)

with open('/tmp/runs/detect/object_detection/20200616_VB_trim.mp4', 'rb') as f:
    image_data = f.read()
    file_obj = io.BytesIO(image_data)
    bucket.upload_fileobj(Fileobj=file_obj, Key=key)
