import argparse
import time
from pathlib import Path

import cv2
import torch
import json
from models.experimental import attempt_load
import os
import io
from pandas import DataFrame
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
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import time_synchronized, TracedModel
from utils.plots import plot_one_box
from tracker.byte_tracker import BYTETracker
stride = None
imgsz=1024
resource = boto3.resource('s3')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def model_fn(model_dir):
    global stride
    device = get_device()
    logger.info(">>> Device is '%s'.." % device)
    model = attempt_load(model_dir + '/best.pt', map_location=torch.device(device))
    logger.info(">>> Model Type!..")
    logger.info(type(model))
    logger.info(">>> Model loaded!..")
    logger.info(model)
    stride = int(model.stride.max())
    model = TracedModel(model, device, imgsz)
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
        local_filename = os.path.join(temp_dir, base_filename)

        # Download the S3 file

        my_bucket = resource.Bucket(bucket_name)
        my_bucket.download_file(key, local_filename)

        ouput_path= detect(local_filename,model,frame_height)
        return json.dumps({"output_path":""})

    except Exception as e:
        logger.error(traceback.format_exc())
        return json.dumps({"Error":traceback.format_exc()})

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

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
    global stride
    augment=False
    agnostic_nms=False
    classes=None
    iou_thres=0.45
    conf_thres=0.25
    save_conf=True
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20=False
    save_img=True

    save_with_object_id=True
    save_txt=True
    save_bbox_dim=True

    tracker = BYTETracker(track_thresh,track_buffer,mot20,match_thresh) # track_thresh, match_thresh, mot20
    track_results = {   'Frame': [],
                        'top':[],
                        'left':[],
                        'width': [],
                        'height':[],
                        'track_id':[]
                    }
    #......................... 
    save_dir = Path(increment_path(Path("runs/detect") / "object_detection", exist_ok=True))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    
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
   
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    device = get_device()

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(video_file, img_size=imgsz, stride=stride)
    half = device!= 'cpu'
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    frame_id = 0

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        frame_id += 1
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
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            dets = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                    dets.append([xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item(), conf.item()])

            # Tracking
            online_targets = tracker.update(np.array(dets), [old_img_w, old_img_h], (img.shape[3], img.shape[2]))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                track_results['Frame']   .append(frame_id)
                track_results['top']     .append(tlwh[0])
                track_results['left']    .append(tlwh[1])
                track_results['width']   .append(tlwh[2])
                track_results['height']  .append(tlwh[3])
                track_results['track_id'].append(tid)

                #vertical = tlwh[2] / tlwh[3] > 1.6
                #if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                #    online_tlwhs.append(tlwh)
                #    online_ids.append(tid)
                #    online_scores.append(t.score)
            # save results
            #track_results.append((frame_id, online_tlwhs, online_ids, online_scores))
            t4 = time_synchronized()
            #print(track_results)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({t4-t3:.1f}ms) {len(online_targets)} Tracked.')
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

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        DataFrame(track_results).to_csv('{save_dir}tracked_data.csv')
    
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == "__main__":
    model=model_fn("/home/ubuntu/yolo7-cctv-deployment-aws/temp")
    feed_data={"s3_path":"s3://lightsketch-models-188775091215/models/20200616_VB_trim.mp4"}
    transform_fn(model,feed_data,"application/video","")
    