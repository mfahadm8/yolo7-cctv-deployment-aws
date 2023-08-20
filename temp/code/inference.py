import argparse
import time
from pathlib import Path

import cv2
import torch
import json
from models.experimental import attempt_load
import os
import io

import json
import logging
import tempfile
import boto3
import cv2
import torch
import torchvision.transforms as transforms
import traceback
# This code will be loaded on each worker separately..

from numpy import random
import numpy as np
from random import randint
from utils.datasets import LoadImages
from utils.general import check_img_size,non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized, TracedModel

from tracker.sort import Sort
client = boto3.client('s3')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def model_fn(model_dir):
    device = get_device()
    logger.info(">>> Device is '%s'.." % device)
    model = attempt_load(model_dir + '/best.pt', map_location=torch.device(device))
    logger.info(">>> Model Type!..")
    logger.info(type(model))
    logger.info(">>> Model loaded!..")
    logger.info(model)
    return model

def transform_fn(model, request_body, content_type, accept):
    try:
        interval = int(os.environ.get('FRAME_INTERVAL', 30))
        frame_width = int(os.environ.get('FRAME_WIDTH', 1024))
        frame_height = int(os.environ.get('FRAME_HEIGHT', 1024))
        batch_size = int(os.environ.get('BATCH_SIZE', 12))
        s3_path_without_prefix = request_body["s3_path"][len("s3://"):]
        # Split the path into bucket name and key
        bucket_name, key = s3_path_without_prefix.split('/', 1)
        base_filename = os.path.basename(key)
        # Create a temporary file in the system's temporary directory
        temp_dir = tempfile.gettempdir()
        local_destination = os.path.join(temp_dir, base_filename)

        # Download the S3 file
        client.download_file(bucket_name, key, local_destination)

        detect(local_destination,model,frame_height)
        all_predictions = []
        # for batch_frames in batch_generator(tfile, frame_width, frame_height, interval, batch_size):
            # batch_inputs = preprocess(batch_frames)  # returns 4D tensor
            # batch_outputs = predict(batch_inputs, model)
            # logger.info(">>> Length of batch predictions: %d" % len(batch_outputs))
            # batch_predictions = postprocess(batch_outputs)
            # all_predictions.extend(batch_predictions)

        # logger.info(">>> Length of final predictions: %d" % len(all_predictions))
        # return json.dumps(all_predictions)
    except Exception as e:
        logger.error(traceback.format_exc())
        return json.dumps({"Error":traceback.format_exc()})

def preprocess(inputs, preprocessor=transforms.ToTensor()):
    outputs = torch.stack([preprocessor(frame) for frame in inputs])
    return outputs
    
def predict(inputs, model):
    logger.info(">>> Invoking model!..")

    with torch.no_grad():
        device = get_device()
        model = model.to(device)
        input_data = inputs.to(device)
        model.eval()
        outputs = model(input_data)

    return outputs

def postprocess(inputs):
    outputs = []
    for inp in inputs:
        logger.info("Postprocessing"+str(inp))
        outputs.append({
            'boxes': inp['boxes'].detach().cpu().numpy().tolist(),
            'labels': inp['labels'].detach().cpu().numpy().tolist(),
            'scores': inp['scores'].detach().cpu().numpy().tolist()
        })
    return outputs

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def batch_generator(tfile, frame_width, frame_height, interval, batch_size):
    cap = cv2.VideoCapture(tfile.name)
    frame_index = 0
    frame_buffer = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            cap.release()
            if frame_buffer:
                yield frame_buffer
            return

        if frame_index % interval == 0:
            frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            frame_buffer.append(frame_resized)

        if len(frame_buffer) == batch_size:
            yield frame_buffer
            frame_buffer.clear()

        frame_index += 1
    else:
        raise Exception("Failed to open video '%s'!.." % tfile.name)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img



def detect(video_file,model,imgsz):
    augment=False
    save_img=True
    save_dir="/tmp/"
    save_with_object_id=True
    save_txt=True
    save_bbox_dim=True
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003 # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    device = get_device()
    model = TracedModel(model, device, imgsz)

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(video_file.name, img_size=imgsz, stride=stride)
    half = device.type != 'cpu'
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #loop over tracks
                for track in tracks:
                    [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
            else: #SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()
            #........................................................
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == "__main__":
    model=model_fn("/home/ubuntu/yolo7-cctv-deployment-aws/temp")
    feed_data={"s3_path":"lightsketch-models-188775091215/models/20200616_VB_trim.mp4"}
    transform_fn(model,feed_data,"application/video","")
    