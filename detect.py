# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync

import io
from PIL import Image
import base64 as b64

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
   
    return im, ratio, (dw, dh)

def loadImage(imgx):
    # Read image
    img0 = imgx
    #img0 = cv2.imread(imgx)  # BGR
    assert img0 is not None

    # Padded resize
    img = letterbox(img0, [640, 640], 32)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0

@torch.no_grad()
def run(weights='best.pt',  # model.pt path(s)
        source='test.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=[640, 640],  # inference size (pixels)
        conf_thres=0.85,  # confidence threshold
        iou_thres=0.85,  # NMS IOU threshold
        max_det=10000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    if half:
        model.half()  # to FP16

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    t0 = time.time()

    img, im0s = loadImage(source)
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    t1 = time_sync()

    pred = model(img, augment=augment, visualize=False)[0]
        
    pred[..., 0] *= imgsz[1]  # x
    pred[..., 1] *= imgsz[0]  # y
    pred[..., 2] *= imgsz[1]  # w
    pred[..., 3] *= imgsz[0]  # h
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t2 = time_sync()
    
    nums = 0

    # Process predictions
    for _ , det in enumerate(pred):
          # detections per image
        nums = len(det)
        s, im0 = '', im0s.copy(),
        s += '%gx%g ' % img.shape[2:]  # print string
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            print(det)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            print(det)
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)

    print(f'{s}Done. ({t2 - t1:.3f}s)')
    
    print(f'Done. ({time.time() - t0:.3f}s)')
    
    im = Image.fromarray(im0)
    rawBytes = io.BytesIO()
    im.save(rawBytes, "JPEG")
    rawBytes.seek(0)

    b64s = b64.b64encode(rawBytes.read()).decode()

    return {"image": str(b64s), "predictions": f"{nums}"}

if __name__ == "__main__":
    print(run())
