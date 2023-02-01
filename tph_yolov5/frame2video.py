from turtle import width
import cv2
import numpy as np
import os
from natsort import natsorted

output_folder = "output/"
path_frames = "runs/detect/exp11"
fps = 7

frames = os.listdir(path_frames)
frames = natsorted(frames)

img_ex = cv2.imread(os.path.join(path_frames,frames[0]))
height, width = img_ex.shape[0:2]
  
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(output_folder + 'video.avi', fourcc, fps, (width, height))

for frame in frames:
    img = cv2.imread(os.path.join(path_frames,frame))
    video.write(img)

video.release()