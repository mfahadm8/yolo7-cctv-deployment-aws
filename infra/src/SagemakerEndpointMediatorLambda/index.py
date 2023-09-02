import json
import os
import json
import boto3
import sagemaker
import datetime
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
import uuid
import pymongo
from urllib.parse import quote_plus

pem_locator ='/opt/python/global-bundle.pem'
LABELS_BUCKET=os.environ.get("LABELS_BUCKET")
VIDEO_BUCKET=os.environ.get("VIDEO_BUCKET")
INPUT_BUCKET=os.environ.get("INPUT_BUCKET")
SAGEMAKER_ENDPOINT_NAME=os.environ.get("SAGEMAKER_ENDPOINT_NAME")
ssm_client=boto3.client("ssm")

sm_session = sagemaker.session.Session()
predictor=Predictor(endpoint_name=SAGEMAKER_ENDPOINT_NAME,sagemaker_session=sm_session,serializer=JSONSerializer())
async_predictor = AsyncPredictor(predictor=predictor)
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
    input_video_uri=None
    if "input_s3_uri" in event:
        input_video_uri=event.get("input_s3_uri","s3://lightsketch-models-188775091215/models/20200616_VB_trim.mp4")
    else:
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
        'output_label_location':  "s3://"+LABELS_BUCKET+"/async-inference/"+inference_id+"/"+input_base_filename+".csv",
        'output_video_location':  "s3://"+VIDEO_BUCKET+"/async-inference/"+inference_id+"/"+input_base_file,
        'inference_id':inference_id
    }

    input_s3_uri=f"s3://{INPUT_BUCKET}/async-inference/input/{inference_id}.json"
    # Call the predict method to send the input data to the endpoint asynchronously
    response = async_predictor.predict_async(data=input_data,input_path=input_s3_uri,inference_id=inference_id)
    update_db(input_data)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'label_uri':input_data["output_label_location"],
            'video_uri':input_data["output_video_location"]
        })
    }
    