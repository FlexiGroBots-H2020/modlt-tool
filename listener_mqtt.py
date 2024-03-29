import argparse
import paho.mqtt.client as mqtt
import os
import json
import base64
import time
from collections import deque

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(userdata['topic'])

def on_message(client, userdata, msg):
    current_time = time.time()
    json_data = json.loads(msg.payload.decode())
    device_id = json_data['device']
    frame_id = json_data['frame']
    init_time = json_data['init_time']
    tracks = json_data['tracks']
    warnings = json_data['warnings']
    distances = json_data['distances']

    inference_delay = current_time - float(init_time)
    print(f"{msg.topic} - Image {frame_id} from device {device_id} written at {time.time()}. Inference delay: {inference_delay:.2f} seconds")
    print(f"Tracks: {tracks}")
    print(f"Distances: {distances}")
    print(f"Warnings: {warnings}")

    if len(userdata['inference_delay_log']) >= 100:
        userdata['inference_delay_log'].popleft()
    userdata['inference_delay_log'].append(inference_delay)

    # Calcular y mostrar la media del retraso de la inferencia de los últimos 100 frames
    average_inference_delay = sum(userdata['inference_delay_log']) / len(userdata['inference_delay_log'])
    print(f"Mean delay last 100 inferences: {average_inference_delay:.2f} s")

    
    # Calculate and print FPS
    last_frame_time = userdata['last_frame_time']
    if last_frame_time is None:
        userdata['last_frame_time'] = current_time
    else:
        time_difference = current_time - last_frame_time
        fps = 1 / time_difference if time_difference > 0 else 0
        print(f"FPS: {fps:.2f}")

        if len(userdata['fps_log']) >= 100:
            userdata['fps_log'].popleft()
        userdata['fps_log'].append(fps)

        mfps = sum(userdata['fps_log']) / len(userdata['fps_log'])
        print(f"MFPS (Mean last 100 FPS): {mfps:.2f}")

        userdata['last_frame_time'] = current_time
    
    
def main(args):
    client_id = "listener_" + args.topic
    client = mqtt.Client(client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(os.getenv('BROKER_ADDRESS'), int(os.getenv('BROKER_PORT')), 60)
    client.username_pw_set(os.getenv('BROKER_USER'), os.getenv('BROKER_PASSWORD'))

    # Include 'last_frame_time' in the userdata dictionary
    userdata = {'topic': args.topic, 'last_frame_time': None, 'fps_log': deque(), 'inference_delay_log': deque()}
    client.user_data_set(userdata)

    client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MQTT subscriber script.')
    parser.add_argument('--topic', type=str, required=True, help='Topic to subscribe to')
    args = parser.parse_args()
    main(args)
