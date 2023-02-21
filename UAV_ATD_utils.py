import argparse
from pathlib import Path
import sys
import os
import numpy as np
import cv2
import torch
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parser_ATD():
    parser = argparse.ArgumentParser("TPH-yolov5 + Strong-SORT Demo!")
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam') 
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    ## TPH-Yolov5
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--show-results', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-img', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide class')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    
    # tracking args
    parser.add_argument('--max_iou_distance', type=float, default=0.7,
                        help='Maximum distance overlap between consecutive frames')
    parser.add_argument('--nms_max_overlap', type=float, default=0.3,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.7,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')
    parser.add_argument('--max_age', type=float, default=50,
                        help='Num of iterations until delete a track')
    parser.add_argument('--n_init', type=float, default=3,
                        help='Num of detections until consider a track as valid')
   
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, classes=None, frame_id=0, fps=0., ids2=None, text_detection=False):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '#{}'.format(int(obj_id))
        if classes is not None:
            id_text = id_text + ':{}'.format(classes[i])
        if scores is not None:
            id_text = id_text + ' {:.2f}'.format(float(scores[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        if text_detection:
            cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                        thickness=text_thickness)
    return im

def xyxy2xywh(x):
    # Convert 1x4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


def xyxy2tlwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]            # x min
    y[:, 1] = x[:, 1]            # y min
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def update_tracks(tracker, frame_count, save_txt, txt_path, save_img, view_img, img, names, thickness=3, info=True):
    if len(tracker.tracks):
        print("[Tracks]", len(tracker.tracks))

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlbr()
        class_num = track.class_num
        bbox = np.round(xyxy)
        class_name = names[int(class_num)]
        if info:
            print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        if save_txt:  # Write to file  

            # Create folder to store output
            if not os.path.exists(txt_path):
                os.makedirs(txt_path)
                
            xywh = xyxy2xywh(bbox)  # normalized xywh
        
            with open(txt_path + '.txt', 'a') as f:
                f.write('frame: {}; track: {}; class: {}; bbox: {}\n'.format(frame_count, track.track_id, class_num,
                                                                              *xywh))

        if save_img or view_img:  # Add bbox to image
            label = f'{class_name} #{track.track_id}'
            plot_one_box(xyxy, img, label=label, color=get_color_for(label), line_thickness=thickness)


def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def distance_between_bboxes(bboxes, image):
    """
    Calculates the distance between all pairs of bounding boxes in a list in function of their sizes and adds lines connecting their center points to the input image with a color code according to the relative distance.

    Parameters:
    bboxes (list): A list of tuples representing the bounding boxes in the format (x, y, width, height).
    image (ndarray): A NumPy array representing the input image with bounding boxes drawn on it.

    Returns:
    list: A list of tuples representing the distances between all pairs of bounding boxes.
    ndarray: A NumPy array representing the input image with lines connecting the center points of the bounding boxes added to it.
    """
    distances = []

    # Loop through all possible pairs of bounding boxes and calculate the distances between them.
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bbox1 = bboxes[i]
            bbox2 = bboxes[j]
            # Call the distance_between_bboxes function to calculate the distance and draw the line.
            distance, image = distance_between_bboxes(bbox1, bbox2, image)
            distances.append(distance)

    return distances, image


def distance_between_2bboxes(bbox1, bbox2, image, dist_thres):
    """
    Calculates the distance between two bounding boxes in function of their sizes and adds a line connecting their center points to the input image with a color code according to the relative distance.

    Parameters:
    bbox1 (tuple): A tuple representing the first bounding box in the format (x, y, width, height).
    bbox2 (tuple): A tuple representing the second bounding box in the format (x, y, width, height).
    image (ndarray): A NumPy array representing the input image with bounding boxes drawn on it.
    dist_thres (int): A int value that set the threshold to consider to objects close to each other.

    Returns:
    float: The distance between the two bounding boxes.
    ndarray: A NumPy array representing the input image with a line connecting the center points of the bounding boxes added to it.
    """
    # Extract the x, y, width, and height values from the bounding boxes.
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the center points of the bounding boxes.
    center1_x = x1 + (w1 / 2)
    center1_y = y1 + (h1 / 2)
    center2_x = x2 + (w2 / 2)
    center2_y = y2 + (h2 / 2)

    # Calculate the distance between the center points of the bounding boxes.
    distance = ((center1_x - center2_x) / ((w1 + w2) / 2))**2 + ((center1_y - center2_y) / ((h1 + h2) / 2))**2

    # Determine the color of the line based on the relative distance between the bounding boxes.
    color = tuple(np.array([0, 255, 0]) + (np.array([0, 0, 255]) - np.array([0, 255, 0])) * (1 - distance/dist_thres))

    # Add a line connecting the center points to the image.
    cv2.line(image, (int(center1_x), int(center1_y)), (int(center2_x), int(center2_y)), color, 3)

    return distance, image


def distance_between_bboxes(bboxes, image, dist_thres=30):
    """
    Calculates the distance between all pairs of bounding boxes in a list in function of their sizes and adds lines connecting their center points to the input image with a color code according to the relative distance.

    Parameters:
    bboxes (list): A list of tuples representing the bounding boxes in the format (x, y, width, height).
    image (ndarray): A NumPy array representing the input image with bounding boxes drawn on it.
    dist_thres (int): A int value that set the threshold to consider to objects close to each other.

    Returns:
    list: A list of tuples representing the distances between all pairs of bounding boxes.
    ndarray: A NumPy array representing the input image with lines connecting the center points of the bounding boxes added to it.
    """
    distances = []

    # Loop through all possible pairs of bounding boxes and calculate the distances between them.
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bbox1 = bboxes[i][0]
            bbox2 = bboxes[j][0]
            id_bbx1 = bboxes[i][1]
            id_bbx2 = bboxes[j][1]
            clss_bbx1 = bboxes[i][2]
            clss_bbx2 = bboxes[j][2]
            # Call the distance_between_bboxes function to calculate the distance and draw the line.
            distance, image = distance_between_2bboxes(bbox1, bbox2, image, dist_thres)
            distances.append((distance, (id_bbx1, id_bbx2), (clss_bbx1, clss_bbx2)))

    return distances, image


def extract_bbox_from_track(tracker):
    list_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        xyxy = track.to_tlwh()
        list_bboxes.append((np.round(xyxy), track.track_id, track.class_num))
    return list_bboxes


def dist_awareness(tracker, distances, names, dist_awr_thres = 10):
    list_msgs = []
    for distant in distances:
        if distant[0] <= dist_awr_thres:
            list_msgs.append(names[int(distant[2][0])] + " " + str(distant[1][0]) + " and " + names[int(distant[2][1])] + " " + str(distant[1][1]) + " are too close")
    return list_msgs