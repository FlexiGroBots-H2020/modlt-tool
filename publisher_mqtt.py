import argparse
import base64
import json
import cv2
import numpy as np
import paho.mqtt.publish as publish
import time
import os
import paho.mqtt.client as mqtt

def on_publish(client, userdata, result):
    print(f"Data published: {result}")

# Función para codificar la imagen en base64
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Función para calcular las nuevas dimensiones
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

# Función para redimensionar la imagen
def resize_image(image, new_dimensions):
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

# Función principal
def main(args):
    client = mqtt.Client("modtl-model")
    client.on_publish = on_publish
    client.username_pw_set(os.getenv('BROKER_USER'), os.getenv('BROKER_PASSWORD'))
    client.connect(os.getenv('BROKER_ADDRESS'), int(os.getenv('BROKER_PORT')))

    # Abre el video
    video = cv2.VideoCapture(args.video_path)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_to_skip = fps // args.fpsp
    frame_id = 0

    # Procesa cada fotograma del video
    max_dimension = 1536
    current_frame = 0
    while video.isOpened():
        start_time = time.time()
        ret, frame = video.read()

        if not ret:
            break

        if current_frame % frames_to_skip == 0:
            read_time = time.time() - start_time
            print(f"Current frame: {current_frame}")
            print(f"Read time: {read_time:.2f}s")

            start_time = time.time()
            new_dimensions = calculate_new_dimensions(frame, max_dimension)
            resized_frame = resize_image(frame, new_dimensions)
            resize_time = time.time() - start_time
            print(f"Resize time: {resize_time:.2f}s")

            start_time = time.time()
            encoded_frame = encode_image(resized_frame)
            encode_time = time.time() - start_time
            print(f"Encode time: {encode_time:.2f}s")

            # Crea el objeto JSON con la imagen y metadatos adicionales
            payload_json = json.dumps({'img': encoded_frame, 'device_id': args.device_id, 'frame_id': frame_id, 'init_time': time.time()})

            # Conéctate al broker MQTT
            start_time = time.time()
            try:
                publish.single("common-apps/modtl-model/input",
                            payload_json,
                            hostname=os.getenv('BROKER_ADDRESS'),
                            port=int(os.getenv('BROKER_PORT')),
                            client_id="modtl-model",
                            auth={"username": os.getenv('BROKER_USER'), "password": os.getenv('BROKER_PASSWORD')})
                publish_time = time.time() - start_time
                print(f"Publish time: {publish_time:.2f}s")
            except Exception as e:
                print(f'error publishing: {e}')

            time.sleep(1)
            frame_id += 1
        current_frame += 1

    # Libera los recursos
    print("Video proccessed")
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video processing script.')
    parser.add_argument('--device_id', type=str, required=True, help='Device ID')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    parser.add_argument('--fpsp', type=int, default=1, help='Number of frames to process per second')
    args = parser.parse_args()
    main(args)
