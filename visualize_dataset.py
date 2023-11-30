
import os
import shutil
# Delete .fiftyone folder 
fiftyone_temp_path = os.path.join(os.path.expanduser('~'), ".fiftyone")
if os.path.exists(fiftyone_temp_path):
    shutil.rmtree(fiftyone_temp_path)
    print("Deleted previous fiftyone session folder: " + fiftyone_temp_path)

import fiftyone as fo

# The directory containing the dataset to import
dataset_dir = "/home/atos/Escritorio/FlexiGroBots/datasets/VisDrone_Tractor_V2"

splits = ["train", "val", "test"]

# The type of the dataset being imported
dataset_type = fo.types.YOLOv5Dataset  # for example

# Import the dataset
dataset = fo.Dataset("VisDrone_Tractor")

for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        split=split,
        tags=split, 
        classes=["Vehicle registration plate"]
)
print(dataset)
fo.pprint(dataset.stats(include_media=True))

session = fo.launch_app(dataset)

# Obtener estadísticas de la distribución de clases
class_counts = dataset.count_values("ground_truth.detections.label")

# Imprimir las estadísticas
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} instances")


# View the dataset's current App config
print(dataset.app_config)