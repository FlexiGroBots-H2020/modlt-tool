'''
This script implements TPH-yoloV5 with Zero-shot tracker to perform ATD in Drone taken videos or sequence of images 
'''

import sys
import os
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
import os.path as osp
import logging
import paho.mqtt.publish as publish

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools_clip import generate_clip_detections as gdet
import clip
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'tph-yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'tph_yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from UAV_ATD_utils import get_fps,parser_ATD, update_tracks, distance_between_bboxes, extract_bbox_from_track, dist_awareness

# Imports TPH-YOLOv5
from tph_yolov5.models.experimental import attempt_load
from tph_yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from tph_yolov5.utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from tph_yolov5.utils.plots import Annotator, colors
from tph_yolov5.utils.torch_utils import load_classifier, select_device, time_sync

from tph_yolov5.utils.augmentations import letterbox


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def main(args):
    # Handle inputs
    source = str(args.source)
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / 'labels' if args.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    if not args.nosave:
        vis_folder = osp.join(save_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    
    # initialize deep sort
    nms_max_overlap = args.nms_max_overlap
    max_cosine_distance = args.max_cosine_distance
    nn_budget = args.nn_budget
    max_age=args.max_age
    n_init=args.n_init
    max_iou_distance=args.max_iou_distance
    model_filename = "models_clip/ViT-B-32.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    model, transform = clip.load(model_filename, device=device, jit=False)
    model.eval()
    encoder = gdet.create_box_encoder(model, transform, batch_size=1, device=device)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
    tracker = Tracker(metric, max_iou_distance, max_age, n_init) # initialize tracker
    
    # Initialize
    device = select_device(args.device)
    args.device = device
    half = args.half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load TPH-yolov5 model
    w = str(args.weights[0] if isinstance(args.weights, list) else args.weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(args.weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        if args.dnn:
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        else:
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
    else: 
        logging.info("TensorFlow has been disabled") # if you want to recover it take it from tph-yolov5 detect
   
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        args.view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
        fps = get_fps(source)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        fps = round(get_fps(source),0)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    current_time = time.localtime()
    frame_id = 0
    visualize = args.visualize
    results = []
    
    #   Processing loop 
    for path, img, im0s, vid_cap, s in dataset:

        if (frame_id%(round(fps/args.pfps,0))==0): # only process some of the frames
            # Load image
            t1 = time_sync()
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            if onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1


            # Inference
            if pt:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(img, augment=args.augment, visualize=visualize)[0]
            elif onnx:
                if args.dnn:
                    net.setInput(img)
                    pred = torch.tensor(net.forward())
                else:
                    pred = 0
            else: 
                logging.info("TensorFlow has been disabled") # if you want to recover it take it from tph-yolov5 detect
            
            t3 = time_sync()
            dt[1] += t3 - t2
            
            # NMS detected bboxes
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
            
            t4 = time_sync()
            dt[2] += t4 - t3 
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                #imc = im0.copy() if args.save_crop else im0  # for save_crop
                #annotator = Annotator(im0, line_width=args.line_thickness, example=str(names))
                    

                if len(det):
                    # Rescale boxes from img_size to im0 size                
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string

                    # Transform bboxes from tlbr to tlwh
                    trans_bboxes = det[:, :4].clone()
                    trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                    bboxes = trans_bboxes[:, :4].cpu()
                    confs = det[:, 4]
                    class_nums = det[:, -1].cpu()
                    classes = class_nums

                    # encode yolo detections and feed to tracker
                    features = encoder(im0, bboxes)
                    detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                        bboxes, confs, classes, features)]

                    # run non-maxima supression
                    boxs = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    class_nums = np.array([d.class_num for d in detections])
                    indices = preprocessing.non_max_suppression(
                        boxs, class_nums, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]

                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)

                    # update tracks
                    list_tracks = update_tracks(tracker, frame_id, False, txt_path, save_img, args.show_results, im0, names)
                    
                    t5 = time_sync()
                    dt[3] += t5 - t4 
                    
                    LOGGER.info(f'{s}Done. YOLO:({t4 - t2:.3f}s), Zero-shot tracker:({t5 - t4:.3f}s)')
                    
                else:
                    LOGGER.info('No detections')
                    
            list_bboxes = extract_bbox_from_track(tracker)
            if list_bboxes != []:
                distances, img_distance = distance_between_bboxes(list_bboxes, im0, dist_thres=20)
                list_dist_awrns = dist_awareness(tracker, distances, names, dist_awr_thres=10)
                LOGGER.info(list_dist_awrns)

                h,w,c = im0.shape

                offset = 10 

                font = cv2.FONT_HERSHEY_SIMPLEX

                for itr, word in enumerate(list_dist_awrns):
                    offset += 100
                    cv2.putText(im0, word, (20, offset), font, 1, (0, 0, 255), 3)
                            
            else:
                distances = []
                img_distance = im0
                list_dist_awrns = []

            # Save txt
            if args.save_txt or args.mqtt_output:
                start_time = time.time()
                dict_out = {"init_time": time.time() ,
                            "device": args.source , 
                            "frame": frame_id, 
                            "tracks": list_tracks, 
                            "warnings": list_dist_awrns, 
                            "distances": str(distances)
                            }
                path_out_txt = txt_path + "_" + str(frame_id) + ".txt"
                with open(path_out_txt, "w+") as file:
                    json.dump(dict_out, file)
                encode_time = time.time() - start_time
                logging.info(f"dict out time: {encode_time:.4f}s")

            
            # Save video (tracking)
            if not args.nosave:
                if not isinstance(vid_writer, cv2.VideoWriter):
                    w, h =  im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
                
            try:
                logging.info(f"frame {frame_id}/{dataset.frames} Done. (detect:{t4 - t2:.3f}s / track:{t5 - t4:.3f}s)")   
            except:
                logging.info(f"frame {frame_id}/{dataset.nf} Done. (detect:{t4 - t2:.3f}s / track:{t5 - t4:.3f}s)") 
            
            if args.mqtt_output:
                if (frame_id%args.pfps==0): #only one per second
                    try:
                        # publish only 1 msg per second
                        mqtt_topic = args.mqtt_topic
                        robot_id = args.robot_id
                        start_time = time.time()
                        mqtt_topic_publish = os.path.join(mqtt_topic, robot_id)
                        client_id = robot_id + str(source)
                        dict_out = json.dumps(dict_out)
                        publish.single(mqtt_topic_publish, 
                                    json.dumps(dict_out), 
                                    hostname=os.getenv('BROKER_ADDRESS'), 
                                    port=int(os.getenv('BROKER_PORT')), 
                                    client_id=client_id, 
                                    auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
                        encode_time = time.time() - start_time
                        logging.info(f"Publish out time: {encode_time:.2f}s")
                    except Exception as e:
                        logging.info(e)
        
        frame_id += 1

            
        logging.info(f"Done") 
    
        vid_writer.release()
        
        
        # write all results to txt
        '''
        if args.save_txt:
            res_file = osp.join(vis_folder, f"{timestamp}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logging.info(f"save results to {res_file}")
        '''

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        logging.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS %.1fms track per image at shape {(1, 3, *imgsz)}' % t)
        if args.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_txt else ''
            logging.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if args.update:
            strip_optimizer(args.weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    args = parser_ATD()
    
    main(args)
