import boto3
import io

s3 = boto3.resource('s3')

bucket_name = 'sm-ball-tracking-output-blobs'
key = 'async-inference/0fbbf919-885f-4d3b-8be9-fd55c89e164a/20200616_VB_trim.mp4'

bucket = s3.Bucket(bucket_name)

with open('/tmp/runs/detect/object_detection/20200616_VB_trim.mp4', 'rb') as f:
    image_data = f.read()
    file_obj = io.BytesIO(image_data)
    bucket.upload_fileobj(Fileobj=file_obj, Key=key)
