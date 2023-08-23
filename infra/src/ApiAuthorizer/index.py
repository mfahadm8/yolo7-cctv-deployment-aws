import boto3
import os

PASSWORD_PARAM_NAME=os.environ.get("PASSWORD_PARAM_NAME")
ssm_client=boto3.client("ssm")

def get_password():
    response = ssm_client.get_parameter(
        Name=PASSWORD_PARAM_NAME,
        WithDecryption=True 
    )

    return response['Parameter']['Value']

def lambda_handler(event, context):
    
    #1 - Log the event
    print('*********** The event is: ***************')
    print(event)
    
    #2 - See if the person's token is valid
    if event['authorizationToken'] == get_password():
        auth = 'Allow'
    else:
        auth = 'Deny'
    
    arnConstruct=event["methodArn"].split(":")
    apiGatewayId=arnConstruct[-1].split("/")[0]
    #3 - Construct and return the response
    authResponse = { "principalId": "abc123", "policyDocument": { "Version": "2012-10-17", "Statement": [{"Action": "execute-api:Invoke", "Resource": [ "arn:aws:execute-api:"+arnConstruct[3]+":"+arnConstruct[4]+":"+apiGatewayId+"/*/*/"], "Effect": auth}] }}
    return authResponse