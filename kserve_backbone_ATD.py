import sys
import kserve
from typing import Dict
import logging
import torch
import time
import json
import os

from kserve_utils import decode_image, encode_im_np2b64str, dict2json
import torch

import sys
sys.path.insert(0, 'tph_yolov5/')

from UAV_ATD_kserve import load_model, init_model, infer

#from mqtt_client import MQTTClient
import paho.mqtt.publish as publish

# constants
ENCODING = 'utf-8'

class Model(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        
        logging.info("Init kserve inference service: %s", name)
        
        logging.info("GPU available: %d" , torch.cuda.is_available())
        
        # Set up MQTT client instance 
        #self.client = MQTTClient(name)
        
        # Define and initialize all needed variables
        try:
            init_model(self)
            logging.info("Model initialized")
        except Exception as e:
            logging.warning("error init model: " + str(e))
        
        # Instance Detic Predictor
        try:
            logging.info("loading model")
            load_model(self)
        except Exception as e:
            logging.warning("error loading model: " + str(e))
        

    def predict(self, request: Dict):
        logging.info("Predict call -------------------------------------------------------")
        
        #logging.info("Payload: %s", request)
        start_time = time.time()
        
        # Extract input variables from request
        img_b64_str = request["img"]
        id = request["device_id"]
        frame = request["frame_id"]
        init_time = request["init_time"]
        
        try:
            im = decode_image(img_b64_str)
        except Exception as e:
            logging.info("Error prepocessing image: {}".format(e))
        
        decode_time = time.time() - start_time
        logging.info(f"Im Decode time: {decode_time:.2f}s")
        
        start_time = time.time()
        
        try:  
            out_img, distances = infer(self, im, id, frame)
            logging.info(str(distances))
        except Exception as e:
            logging.info("Error processing image: {}".format(e))
            
        infer_time = time.time() - start_time
        logging.info(f"Inference time: {infer_time:.2f}s")
        
        # Encode out imgs
        start_time = time.time()
        out_img_b64_str = encode_im_np2b64str(out_img)
        #dict_out = {"device":id , "frame":frame , "im_detection":out_img_b64_str}
        dict_out = {"init_time":init_time ,"device":id , "frame":frame, "im_detection":out_img_b64_str}
        
        #logging.info(dict_out)
        encode_time = time.time() - start_time
        logging.info(f"Im Encode time: {encode_time:.2f}s")
        logging.info("Image processed")
        
        # Connect to the MQTT broker 
        #self.client.connect()
        # Publish a message 
        #self.client.publish("common-apps/{}/output".format(self.name), json.dumps(dict_out)) 
        start_time = time.time()
        publish.single("common-apps/modtl-model/output", 
                       json.dumps(dict_out), 
                       hostname=os.getenv('BROKER_ADDRESS'), 
                       port=int(os.getenv('BROKER_PORT')), 
                       client_id=self.name, 
                       auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
        encode_time = time.time() - start_time
        logging.info(f"Publish out time: {encode_time:.2f}s")
        # Disconnect from the MQTT broker 
        #self.client.disconnect()

        return {}


if __name__ == "__main__":
    model = Model("modtl-model")
    kserve.KFServer().start([model])

