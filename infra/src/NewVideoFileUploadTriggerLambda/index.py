import json
import os
import boto3 
import urllib
lambda_client = boto3.client('lambda')

def lambda_handler(events, context):
    print(events)
    if "Records" in events:
        for record in events["Records"]:
            bucket_name=urllib.parse.unquote(record["s3"]["bucket"]["name"])
            s3_file_path=urllib.parse.unquote(record["s3"]["object"]["key"])
            if s3_file_path.endswith(".mp4") or s3_file_path.endswith(".MOV"):
                s3_input_uri=f"s3://{bucket_name}/{s3_file_path}"
                input_data = {
                    'input_s3_uri': s3_input_uri,
                }

                target_lambda_function_name = "SagemakerEndpointMediatorLambda"

                response = lambda_client.invoke(
                    FunctionName=target_lambda_function_name,
                    InvocationType='RequestResponse', 
                    Payload=json.dumps(input_data)  
                )
                response_payload = json.loads(response['Payload'].read().decode("utf-8"))

                print ("response_payload: {}".format(response_payload))
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps('Not an .mp4 or .MOV file, ignoring it')
                }
                        
    return {
        'statusCode': 200,
        'body': json.dumps('Sagemaker Mediator Lambda executed successfully!')
    }