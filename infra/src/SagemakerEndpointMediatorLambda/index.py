import json
import os
import json
import boto3

LABELS_BUCKET=os.environ.get("LABELS_BUCKET")
VIDEO_BUCKET=os.environ.get("VIDEO_BUCKET")
SAGEMAKER_ENDPOINT=os.environ.get("SAGEMAKER_ENDPOINT")
boto_session = boto3.session.Session()
sm_runtime = boto_session.client("sagemaker-runtime")

def lambda_handler(event, context):
    print(event)
    body=json.loads(event.get("body"))
    input_uri=body.get("input_s3_uri","s3://lightsketch-models-188775091215/models/20200616_VB_trim.mp4")
    s3_input_path_without_prefix = output_location[len("s3://"):]
    input_bucket_name, input_key = s3_input_path_without_prefix.split('/', 1)
    input_base_file = os.path.basename(input_key)
    input_base_filename= os.path.splitext(input_base_file)[0]
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=SAGEMAKER_ENDPOINT, 
        InputLocation=input_uri
        )
    output_location = response['OutputLocation']
    s3_output_path_without_prefix = output_location[len("s3://"):]
    output_bucket_name, output_key = s3_output_path_without_prefix.split('/', 1)
    output_base_file = os.path.basename(output_key)
    output_base_filename= os.path.splitext(output_base_file)[0]
    return {
        'statusCode': 200,
        'body': {
            'label_uri': "s3://"+LABELS_BUCKET+"/"+output_base_filename+"/"+input_base_filename+".csv",
            'video_uri': "s3://"+VIDEO_BUCKET+"/"+output_base_filename+"/"+input_base_file
        }
    }