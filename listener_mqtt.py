import argparse
import paho.mqtt.client as mqtt
import os
import json
import base64
import time

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(userdata['topic'])

def on_message(client, userdata, msg):
    json_data = json.loads(msg.payload.decode())
    img_base64_str = json_data['im_detection']
    device_id = json_data['device']
    frame_id = json_data['frame']
    init_time = json_data['init_time']

    output_folder = "output/" + device_id
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, f'decoded_image_{frame_id}.jpg')

    img = base64.b64decode(img_base64_str)
    with open(output_filename, "wb") as image_file:
        image_file.write(img)

    inference_delay = time.time() - init_time
    print(f"{msg.topic} - Image {frame_id} from device {device_id} written at {time.time()}. Inference delay: {inference_delay:.2f} seconds")

def main(args):
    client = mqtt.Client("listener_mario")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(os.getenv('BROKER_ADDRESS'), int(os.getenv('BROKER_PORT')), 60)
    client.username_pw_set(os.getenv('BROKER_USER'), os.getenv('BROKER_PASSWORD'))

    client.user_data_set({'topic': args.topic})

    client.loop_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MQTT subscriber script.')
    parser.add_argument('--topic', type=str, required=True, help='Topic to subscribe to')
    args = parser.parse_args()
    main(args)
