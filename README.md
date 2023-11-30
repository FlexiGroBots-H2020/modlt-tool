MOBILE OBJECT DETECTION TRACKING and LOCALIZATION tool

placeholder:
./models_clip/ViT-B-16.pt
./models_clip/ViT-B-32.pt
./input
./yolov5s.pt

*SETTINGS*
Here is a list of the arguments and options you can use:

--source: Specifies the source of the images. It can be a file, directory, URL, glob, or 0 for webcam. Default is ROOT / 'data/images'.

--device: Specifies the CUDA device to be used (e.g., '0' or '0,1,2,3'). Leave blank for CPU.

--data: (Optional) Path to the dataset.yaml file. Default is 'tph_yolov5/data/dataset_tractor.yaml'.

--weights: Path(s) to the model or models to be used. Default is './tph_yolov5/models/best_tractorA.pt'.

--imgsz, --img, --img-size: Image inference size h, w. Default is [1536].

--conf-thres: Confidence threshold. Default is 0.15.

--iou-thres: NMS IoU threshold. Default is 0.1.

--max-det: Maximum detections per image. Default is 100.

--txt_path: (Optional) Path to store text data. Default is ''.

--show-results: If set, the results will be displayed.

--save-txt: If set, results will be saved to *.txt files.

--save-img: If set, results will be saved as images. Default is True.

--save-conf: If set, save confidences in --save-txt labels.

--save-crop: If set, save cropped prediction boxes.

--nosave: If set, images/videos will not be saved.

--classes: Filter by class, e.g. --classes 0, or --classes 0 2 3.

--agnostic-nms: If set, class-agnostic NMS will be used.

--augment: If set, augmented inference will be used.

--visualize: If set, features will be visualized.

--update: If set, all models will be updated.

--project: Save results to project/name. Default is ROOT / 'runs/detect'.

--name: Save results to project/name. Default is exp.

--exist-ok: If set, existing project/name is ok, do not increment.

--line-thickness: Bounding box thickness (in pixels). Default is 3.

--hide-labels: If set, labels will be hidden.

--hide-conf: If set, confidences will be hidden.

--hide-class: If set, class will be hidden.

--half: If set, use FP16 half-precision inference.

--dnn: If set, use OpenCV DNN for ONNX inference.

Tracking Arguments
--max_iou_distance: Maximum distance overlap between consecutive frames. Default is 0.95.

--nms_max_overlap: Non-maxima suppression threshold: Maximum detection overlap. Default is 0.3.

--max_cosine_distance: Gating threshold for cosine distance metric (object appearance). Default is 0.9.

--nn_budget: Maximum size of the appearance descriptors gallery. If None, no budget is enforced. Default is None.

--max_age: Number of iterations until a track is deleted. Default is 50.

--n_init: Number of detections until a track is considered valid. Default is 3.

*INFERENCE SERVICE*
docker build -t ghcr.io/flexigrobots-h2020/modtl-tool:v0 -f Dockerfile .

*LOCAL*
docker pull ghcr.io/flexigrobots-h2020/modtl-local-tool:v0

docker run -it -v "$(pwd)":/wd/shared --name asaw ghcr.io/flexigrobots-h2020/modtl-local-tool:v0 --source shared/input/DJI_0455_recorte.mp4 --data tph_yolov5/data/dataset_tractor.yaml --weights tph_yolov5/models/best_tractorA.pt --img 1920 --save-txt --iou-thres 0.001 --n_init 25 --max_cosine_distance 0.99 --max_iou_distance 0.99 --conf-thres 0.2 --project shared/output


TROUBLESHOOTING

pip install fiftyone-db-ubuntu2204
