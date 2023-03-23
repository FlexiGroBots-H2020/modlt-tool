# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt /wd
RUN pip install -r requirements.txt

COPY UAV_ATD.py /wd
COPY UAV_ATD_utils.py /wd
COPY deep_sort/ /wd/deep_sort/
COPY tools_clip/ /wd/tools_clip/
COPY tph_yolov5/ /wd/tph_yolov5/
COPY yolov5s.pt /wd
COPY models_clip/ /wd/models_clip

ADD https://ultralytics.com/assets/Arial.ttf /home/jovyan/.config/Ultralytics/
RUN chmod -R 777 /home/jovyan/.config/Ultralytics/Arial.ttf

RUN chmod -R 777 /wd

USER jovyan
#RUN mkdir -p /wd/outputs/face_detection_cache/FaceDetector/


ENTRYPOINT ["python", "-u", "UAV_ATD.py"]
