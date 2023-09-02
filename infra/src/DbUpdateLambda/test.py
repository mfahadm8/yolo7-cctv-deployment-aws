
from index import lambda_handler
event={"input_location": "s3://sm-ball-tracking-input-blobs/20200616_VB_trim.mp4", "output_label_location": "s3://sm-ball-tracking-output-labels/async-inference/e37255c7-eba6-4cc8-bff9-863c9b11212a/20200616_VB_trim.csv", "output_video_location": "s3://sm-ball-tracking-output-blobs/async-inference/e37255c7-eba6-4cc8-bff9-863c9b11212a/20200616_VB_trim.mp4", "inference_id": "e37255c7-eba6-4cc8-bff9-863c9b11212a"}
lambda_handler(event,{})