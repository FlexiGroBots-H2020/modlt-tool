import paho.mqtt.client as mqtt
import os
import cv2
import numpy as np
import piexif
import time
import logging

# Configurar el nivel de registro y el formato de salida
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def on_message(client, userdata, msg):
    start_time = time.time()
    logging.info("Recibiendo imagen con metadatos")
    
    logging.info("Extrayendo metadatos EXIF")
    exif_data = piexif.load(msg.payload)
    logging.info("Metadatos extraídos en %.2f segundos", time.time() - start_time)
    print_specified_metadata(exif_data)
    
    start_time = time.time()
    # Eliminar los metadatos EXIF de la imagen
    exif_bytes = piexif.dump(exif_data)
    img_without_metadata = msg.payload[len(exif_bytes):]
    img_data = np.frombuffer(img_without_metadata, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    #cv2.imwrite("recibida.jpg", img)
    #logging.info("Imagen guardada como recibida.jpg")

    logging.info("Imagen recibida en %.2f segundos", time.time() - start_time)

def print_specified_metadata(exif_data):
    if not exif_data or "0th" not in exif_data:
        print("No se encontraron metadatos EXIF en la imagen.")
        return

    specified_tags = [piexif.ImageIFD.Artist, piexif.ImageIFD.ImageID, piexif.ImageIFD.DateTime]

    for tag in specified_tags:
        tag_name = piexif.TAGS["0th"][tag]["name"]
        if tag in exif_data["0th"]:
            tag_value = exif_data["0th"][tag]
            if isinstance(tag_value, bytes):
                tag_value = tag_value.decode("utf-8")
            print(f"{tag_name} ({tag}): {tag_value}")
        else:
            print(f"{tag_name} ({tag}): No encontrado en la imagen")

client = mqtt.Client()
client.on_message = on_message

client.username_pw_set(os.getenv('BROKER_USER'), os.getenv('BROKER_PASSWORD'))
client.connect(os.getenv('BROKER_ADDRESS'), int(os.getenv('BROKER_PORT')), 60)

client.subscribe("common-apps/modtl-model/testing")

client.loop_forever()

