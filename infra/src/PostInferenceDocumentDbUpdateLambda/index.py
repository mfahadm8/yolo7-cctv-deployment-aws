import json
import os
import json
import datetime
import urllib

INPUT_VIDEOS_BUCKET=

def lambda_handler(events, context):
    print(events)
    if "Records" in events:
        for record in events["Records"]:
            bucket_name=urllib.parse.unquote(record["s3"]["bucket"]["name"])
            s3_file_path=urllib.parse.unquote(record["s3"]["object"]["key"])
            inference_id, input_key = s3_file_path.split('/', 1)
            input_base_file = os.path.basename(s3_file_path)
            input_base_filename= os.path.splitext(input_base_file)[0]

            input_data = {
                'timestamp': int(datetime.datetime.now().timestamp()*1000),
                'input_video_location': f"s3://{bucket_name}",
                'output_label_location':  "s3://"+LABELS_BUCKET+"/async-inference/"+inference_id+"/"+input_base_filename+".csv",
                'output_video_location':  "s3://"+VIDEO_BUCKET+"/async-inference/"+inference_id+"/"+input_base_file
            }
