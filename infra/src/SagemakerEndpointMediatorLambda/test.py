import os

os.environ["INPUT_BUCKET"]="sm-ball-tracking-inputs"
os.environ["VIDEO_BUCKET"]="sm-ball-tracking-output-blobs"
os.environ["SAGEMAKER_ENDPOINT_NAME"]="ball-tracking-v7"
os.environ["LABELS_BUCKET"]="sm-ball-tracking-output-labels"

from index import lambda_handler


event={'input_s3_uri': 's3://sm-ball-tracking-input-blobs/20200616_VB_trim.mp4'} 
lambda_handler(event,{})