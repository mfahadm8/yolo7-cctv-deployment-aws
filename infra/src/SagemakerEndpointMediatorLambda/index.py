import json
import os
import json
import boto3
import sagemaker
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
import uuid

LABELS_BUCKET=os.environ.get("LABELS_BUCKET")
VIDEO_BUCKET=os.environ.get("VIDEO_BUCKET")
SAGEMAKER_ENDPOINT_NAME=os.environ.get("SAGEMAKER_ENDPOINT_NAME")
sm_session = sagemaker.session.Session()
predictor=Predictor(endpoint_name=SAGEMAKER_ENDPOINT_NAME,sagemaker_session=sm_session,serializer=JSONSerializer())
async_predictor = AsyncPredictor(predictor=predictor)

def lambda_handler(event, context):
    print(event)
    body=json.loads(event.get("body"))
    input_video_uri=body.get("input_s3_uri","s3://lightsketch-models-188775091215/models/20200616_VB_trim.mp4")
    s3_input_path_without_prefix = input_video_uri[len("s3://"):]
    input_bucket_name, input_key = s3_input_path_without_prefix.split('/', 1)
    input_base_file = os.path.basename(input_key)
    input_base_filename= os.path.splitext(input_base_file)[0]
    inference_id=str(uuid.uuid4())
    # Prepare your custom input data as a dictionary
    input_data = {
        'input_location': input_video_uri,
        'output_label_location':  "s3://"+LABELS_BUCKET+"/"+inference_id+"/"+input_base_filename+".csv",
        'output_video_location':  "s3://"+VIDEO_BUCKET+"/"+inference_id+"/"+input_base_file
    }

    input_s3_uri=f"s3://{bucket}/{prefix}/input/{inference_id}.json"
    # Call the predict method to send the input data to the endpoint asynchronously
    response = async_predictor.predict_async(data=input_data,input_path=input_s3_uri,inference_id=inference_id)
    
    return {
        'statusCode': 200,
        'body': {
            'label_uri':input_data["output_label_location"],
            'video_uri':input_data["output_video_location"]
        }
    }