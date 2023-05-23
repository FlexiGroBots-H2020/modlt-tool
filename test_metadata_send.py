import cv2
import numpy as np
import logging
import piexif
import time
import paho.mqtt.client as mqtt
import os
import paho.mqtt.publish as publish

# Configurar el nivel de registro y el formato de salida
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def on_connect(client, userdata, flags, rc):
    logging.info("Conectado al broker MQTT con resultado: %s", str(rc))

def add_metadata_to_image(input_image_path, metadata):
    start_time = time.time()
    logging.info("Abriendo imagen: %s", input_image_path)
    img = cv2.imread(input_image_path)
    logging.info("Imagen shape {}".format(img.shape))
    logging.info("Imagen abierta en %.2f segundos", time.time() - start_time)

    start_time = time.time()
    logging.info("Convirtiendo metadatos a formato EXIF")
    exif_dict = {"0th": metadata}
    exif_bytes = piexif.dump(exif_dict)
    logging.info("Metadatos convertidos en %.2f segundos", time.time() - start_time)

    start_time = time.time()
    logging.info("Añadiendo metadatos EXIF a la imagen")
    img_encoded, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_with_metadata = bytearray(exif_bytes) + buf.tobytes()

    logging.info("Imagen procesada con metadatos en %.2f segundos", time.time() - start_time)

    return img_with_metadata

# Ejemplo de uso:
input_image_path = "input/example.png"

metadata = {
    piexif.ImageIFD.Artist: "drone_A",
    piexif.ImageIFD.ImageID: "1",
    piexif.ImageIFD.DateTime: str(time.time())
}

img_with_metadata = add_metadata_to_image(input_image_path, metadata)

# Publicar la imagen en el tema MQTT
start_time = time.time()
publish.single("common-apps/modtl-model/testing", 
                img_with_metadata, 
                hostname=os.getenv('BROKER_ADDRESS'), 
                port=int(os.getenv('BROKER_PORT')), 
                client_id="testing_client_sender", 
                auth = {"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')} )
encode_time = time.time() - start_time
logging.info(f"Publish out time: {encode_time:.2f}s")


