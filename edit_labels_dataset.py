import os
import shutil
import time

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

dataset.persistent = True

for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        split=split,
        tags=split
    )

print(dataset)
fo.pprint(dataset.stats(include_media=True))

# Obtener estadísticas de la distribución de clases
class_counts = dataset.count_values("ground_truth.detections.label")

# Imprimir las estadísticas
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} instances")

# Explore your dataset and select/tag a subset for annotation
session = fo.launch_app(dataset)

# Define a unique annotation key
anno_key = "labelstudio_annotation"

# Number of images to process at a time
batch_size = 50
num_images = len(dataset)
num_batches = num_images // batch_size + 1

for batch in range(num_batches):
    start = batch * batch_size
    end = min(start + batch_size, num_images)
    print(f"Processing images {start}-{end} of {num_images}")

    # Select a subset of images
    selected_view = dataset.skip(start).limit(batch_size)

    # Launch the annotation interface
    selected_view.annotate(anno_key, backend="labelstudio", label_field="ground_truth", label_type="detections", classes= ["tractor", "people", "car", "van"], launch_editor=True, url="http://localhost:8080", api_key="3e7f5bdceeab5ae115d428565849511cd99131fe")

    # Load the annotations into the dataset
    selected_view.load_annotations(anno_key)

    # Save the annotations
    selected_view.save_annotation(anno_key)

# Print information about the annotations
print(dataset.get_annotation_info(anno_key))

# View the dataset's current App config
print(dataset.app_config)

# Launch the app
session = fo.launch_app(dataset)
