import json
import boto3
import datetime

dynamodb = boto3.resource('dynamodb')
table_name = 'Tracked'
table = dynamodb.Table(table_name)

def update_db(input_data):
    timestamp = int(datetime.datetime.utcnow().timestamp() * 1000)
    input_data['timestamp'] = timestamp

    response = table.put_item(
        Item=input_data
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print("Document inserted successfully.")
        print("Inserted Document ID:", input_data['inference_id'])
    else:
        print("Document insertion failed.")

def mark_complete_db_item(inference_id):
    response = table.update_item(
        Key={'inference_id': inference_id},
        UpdateExpression="SET #s = :status",
        ExpressionAttributeValues={":status": "Completed"},
        ExpressionAttributeNames={"#s": "status"}
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print("Document updated successfully.")
    else:
        print("Update was not acknowledged by the server.")

def lambda_handler(event, context):
    print(event)
    if 'requestType' in event:
        inference_id = event['inference_id']
        mark_complete_db_item(inference_id)
    else:
        update_db(event)
    return {
        'statusCode': 200,
        'body': "Db Update Successful"
    }
