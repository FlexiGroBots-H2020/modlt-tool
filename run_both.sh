#!/bin/bash
python publisher_mqtt.py --device_id drone_B --input_path input/recorte_twizy.mp4 --topic common-apps/modtl-model/input --fpsp 4 &
python publisher_mqtt.py --device_id drone_C --input_path input/recorte_twizy.mp4 --topic common-apps/modtl-model/input --fpsp 4 &
wait