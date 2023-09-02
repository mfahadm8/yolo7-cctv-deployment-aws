import json
import os
import json
import boto3
import datetime

import pymongo
lambda_client = boto3.client('lambda')
pem_locator ='/opt/python/global-bundle.pem'
db_client=None
def get_credentials():
    """Retrieve credentials from the Secrets Manager service."""
    boto_session = boto3.session.Session()
    try:
        secrets_client = boto_session.client(service_name='secretsmanager', region_name=boto_session.region_name)
        secret_value = secrets_client.get_secret_value(SecretId="DocDBSecret")
        secret = secret_value['SecretString']
        secret_json = json.loads(secret)
        username = secret_json['username']
        password = secret_json['password']
        host = secret_json['host']
        port = secret_json['port']
        return (username, password, host, port)
    except Exception as ex:
        raise
## DOCUMENTDB CONNECTION
def get_db_client():
    # Use a global variable so Lambda can reuse the persisted client on future invocations
    global db_client
    if db_client is None:
        try:
            # Retrieve Amazon DocumentDB credentials
            (secret_username, secret_password, docdb_host, docdb_port) = get_credentials()
            db_client = pymongo.MongoClient(
                    host=docdb_host,
                    port=docdb_port,
                    tls=True,
                    retryWrites=False,
                    tlsCAFile=pem_locator,
                    username=secret_username,
                    password=secret_password,
                    authSource='admin')
            print('Initialized DB Client!')
        except Exception as e:
            print('Failed to create new DocumentDB client: {}'.format(e))
            raise
    return db_client


def update_db(input_data):

    # Define the data you want to insert
    input_data['timestamp'] = int(datetime.datetime.utcnow().timestamp()*1000)

    # Connect to the MongoDB with SSL
    client = get_db_client()
    # Select the database and collection
    db = client["db"]
    collection = db["Tracked"]

    # Insert the data into the collection
    inserted_document = collection.insert_one(input_data)

    # Check if the insertion was successful
    if inserted_document.acknowledged:
        print("Document inserted successfully.")
        print("Inserted Document ID:", inserted_document.inserted_id)
    else:
        print("Document insertion failed.")
    
def lambda_handler(event, context):
    print(event)
    update_db(event)
    return {
        'statusCode': 200,
        'body': json.dumps({
            "Db Update Successful"
        })
    }
    