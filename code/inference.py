import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pandas import DataFrame
import json

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.byte_tracker import BYTETracker
import os
import io

import json
import logging
import tempfile

import cv2
import torch
import torchvision.transforms as transforms

# This code will be loaded on each worker separately..
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
    interval = int(os.environ.get('FRAME_INTERVAL', 30))
    frame_width = int(os.environ.get('FRAME_WIDTH', 1024))
    frame_height = int(os.environ.get('FRAME_HEIGHT', 1024))
    batch_size = int(os.environ.get('BATCH_SIZE', 24))

    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    all_predictions = []

    for batch_frames in batch_generator(tfile, frame_width, frame_height, interval, batch_size):
        batch_inputs = preprocess(batch_frames)  # returns 4D tensor
        batch_outputs = predict(batch_inputs, model)
        logger.info(">>> Length of batch predictions: %d" % len(batch_outputs))
        batch_predictions = postprocess(batch_outputs)
        all_predictions.extend(batch_predictions)
    
    logger.info(">>> Length of final predictions: %d" % len(all_predictions))
    return json.dumps(all_predictions)

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


# def detect(save_img=False):
#     source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
#     save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://', 'https://'))

#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Initialize
#     set_logging()
#     device = select_device(opt.device)
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Initialize Tracker
#     tracker = BYTETracker(opt) # track_thresh, match_thresh, mot20
#     track_results = {   'Frame': [],
#                         'top':[],
#                         'left':[],
#                         'width': [],
#                         'height':[],
#                         'track_id':[]
#                     }
        
#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size

#     if trace:
#         model = TracedModel(model, device, opt.img_size)

#     if half:
#         model.half()  # to FP16

#      # Set Dataloader
#     vid_path, vid_writer = None, None
#     if webcam:
#         view_img = check_imshow()
#         cudnn.benchmark = True  # set True to speed up constant image size inference
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride)

#     # Get names and colors
#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     old_img_w = old_img_h = imgsz
#     old_img_b = 1
    
#     frame_id = 0

#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         frame_id += 1
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)

#         # Warmup
#         if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
#             old_img_b = img.shape[0]
#             old_img_h = img.shape[2]
#             old_img_w = img.shape[3]
#             for i in range(3):
#                 model(img, augment=opt.augment)[0]

#         # Inference
#         t1 = time_synchronized()
#         with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#             pred = model(img, augment=opt.augment)[0]
#         t2 = time_synchronized()

#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#         t3 = time_synchronized()

#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#             if webcam:  # batch_size >= 1
#                 p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
#             else:
#                 p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             dets = []
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
#                         with open(txt_path + '.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                     if save_img or view_img:  # Add bbox to image
#                         label = f'{names[int(cls)]} {conf:.2f}'
#                         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    
#                     dets.append([xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item(), conf.item()])

#             # Tracking
#             online_targets = tracker.update(np.array(dets), [old_img_w, old_img_h], (img.shape[3], img.shape[2]))
#             online_tlwhs = []
#             online_ids = []
#             online_scores = []
#             for t in online_targets:
#                 tlwh = t.tlwh
#                 tid = t.track_id
#                 track_results['Frame']   .append(frame_id)
#                 track_results['top']     .append(tlwh[0])
#                 track_results['left']    .append(tlwh[1])
#                 track_results['width']   .append(tlwh[2])
#                 track_results['height']  .append(tlwh[3])
#                 track_results['track_id'].append(tid)

#                 #vertical = tlwh[2] / tlwh[3] > 1.6
#                 #if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
#                 #    online_tlwhs.append(tlwh)
#                 #    online_ids.append(tid)
#                 #    online_scores.append(t.score)
#             # save results
#             #track_results.append((frame_id, online_tlwhs, online_ids, online_scores))
#             t4 = time_synchronized()
#             #print(track_results)

#             # Print time (inference + NMS)
#             print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({t4-t3:.1f}ms) {len(online_targets)} Tracked.')

#             # Stream results
#             if view_img:
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                     print(f" The image with the result is saved in: {save_path}")
#                 else:  # 'video' or 'stream'
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                             save_path += '.mp4'
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(im0)

#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         print(f"Results saved to {save_dir}{s}")
#         DataFrame(track_results).to_csv('{save_dir}tracked_data.csv')
    
#     print(f'Done. ({time.time() - t0:.3f}s)')

