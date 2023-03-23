import sys
import kserve
from typing import Dict
import logging
import torch

from kserve_utils import decode_im_b642np, encode_im_np2b64str, dict2json
import torch

import sys
sys.path.insert(0, 'tph_yolov5/')

from UAV_ATD_kserve import load_model, init_model, infer

# constants
ENCODING = 'utf-8'

class Model(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        
        logging.info("Init kserve inference service: %s", name)
        
        logging.info("GPU available: %d" , torch.cuda.is_available())
        
        # Define and initialize all needed variables
        try:
            init_model(self)
            logging.info("Model loaded")
        except Exception as e:
            logging.warning("error init model: " + e)
        

    def load(self):
        # Instance Detic Predictor
        try:
            load_model(self)
        except Exception as e:
            logging.warning("error loading model: " + e)
        

    def predict(self, request: Dict):
        
        #logging.info("Payload: %s", request)
        
        # Extract input variables from request
        img_b64_str = request["img"]
        id = request["device_id"]
        frame = request["frame_id"]
        
        try:
            im = decode_im_b642np(img_b64_str)
        except Exception as e:
            logging.info("Error prepocessing image: {}".format(e))
        
        try:  
            out_img, distances = infer(self, im, id, frame)
            logging.info(str(distances))
        except Exception as e:
            logging.info("Error processing image: {}".format(e))
    
        out_img_b64_str = encode_im_np2b64str(out_img)
        
        dict_out = {"device":id ,"frame":frame ,"im_detection":out_img_b64_str}
        
        #logging.info(dict_out)
        logging.info("Image processed")

        return dict2json(dict_out)


if __name__ == "__main__":
    model = Model("modtl-model")
    kserve.KFServer().start([model])

