from mqtt_client import MQTTClient
import cv2
import numpy as np
import json
import argparse
import time
from kserve_utils import NumpyEncoder 
import base64
import os 
import paho.mqtt.publish as publish
import io
from PIL import Image
from kserve_utils import decode_im_b642np_pil

def calculate_new_dimensions(image, max_dimension):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)

    return (new_width, new_height)

def resize_image(image, new_dimensions):
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


def decode_image(encoded_image_data):
    image_data = base64.b64decode(encoded_image_data)
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # resize and change channel order
    new_dimensions = calculate_new_dimensions(image, 1536)

    # Redimensiona la imagen
    img_np = resize_image(image, new_dimensions)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return img_np
    


# Argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Codificar una imagen y enviarla a un broker MQTT')
parser.add_argument('imagen', default="example.png",type=str, help='ruta de la imagen a codificar')
parser.add_argument('--device_id', type=str, default='mi_device', help='identificador del dispositivo')
args = parser.parse_args()


# Set up MQTT client instance 
name="modtl-model"
#start_time = time.time()
#client = MQTTClient(name)
#delayed_time = time.time() - start_time 
#print("Tiempo de instanciación MQTT client: {}".format(delayed_time)) 

# Leer imagen
imagen = cv2.imread(args.imagen)
imagen = cv2.resize(imagen,(1536,768))

# Codificar imagen en formato JPEG
start_time = time.time()
#encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
_, imagen_codificada = cv2.imencode('.jpg', imagen)
delayed_time = time.time() - start_time 
print("Tiempo de cv2.imencode: {}".format(delayed_time)) 

# Convertir imagen codificada a bytes
start_time = time.time()
img_b64_bytes = base64.b64encode(imagen_codificada)
img_b64_str = img_b64_bytes.decode("utf-8")
delayed_time = time.time() - start_time 
print("Tiempo de b64: {}".format(delayed_time)) 

# Crear payload JSON con la imagen codificada
#payload = {
#    'device_id': args.device_id,
#    'imagen': imagen_bytes
#}

# Convertir payload a JSON
#start_time = time.time()
#with open("numpy.json", "w") as outfile:
#    payload_json = json.dumps({'image': imagen_codificada, 'device_id': args.device_id}, cls=NumpyEncoder)
#    json.dump(payload_json, outfile)
#delayed_time = time.time() - start_time 
#print("Tiempo de pasar a json Numpy: {}".format(delayed_time)) 
#print("Tamaño json numpy: {}".format(os.path.getsize("numpy.json"))) 

# Convertir payload a JSON
start_time = time.time()
with open("b64.json", "w") as outfile:
    payload_json = json.dumps({'img': img_b64_str, 'device_id': args.device_id, 'frame_id': "0", 'init_time': time.time()})
    json.dump(payload_json, outfile)
delayed_time = time.time() - start_time 
print("Tiempo de pasar a json b64: {}".format(delayed_time)) 
print("Tamaño json b64: {}".format(os.path.getsize("b64.json")))
 
# testing decode PIL
start_time = time.time()
im=decode_im_b642np_pil(img_b64_str)
delayed_time = time.time() - start_time 
print("Tiempo decodificar pil: {}".format(delayed_time)) 

# testing decode cv2
start_time = time.time()
im=decode_image(img_b64_str)
delayed_time = time.time() - start_time 
print("Tiempo decodificar cv2: {}".format(delayed_time)) 
 

# Connect to the MQTT broker 
start_time = time.time()
publish.single("common-apps/modtl-model/input", 
                payload_json, 
                hostname=os.getenv('BROKER_ADDRESS'), 
                port=int(os.getenv('BROKER_PORT')), 
                client_id=name, 
                auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
encode_time = time.time() - start_time
print(f"Publish out time: {encode_time:.2f}s")

