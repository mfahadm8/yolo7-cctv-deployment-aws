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
            input_data = {
                'requestType':'PostInferenceDBUpdate',
                'inference_id': s3_file_path.split("/")[1],
            }

            target_lambda_function_name = "DbUpdateLambda"

            response = lambda_client.invoke(
                FunctionName=target_lambda_function_name,
                InvocationType='RequestResponse', 
                Payload=json.dumps(input_data)  
            )
            response_payload = json.loads(response['Payload'].read().decode("utf-8"))

            print ("response_payload: {}".format(response_payload))
                        
    return {
        'statusCode': 200,
        'body': json.dumps('DB Updated successfully!')
    }