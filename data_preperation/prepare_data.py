import os
import shutil

input_data_dir = "data"

output_data_dir = "../data"
train_dir = os.path.join(output_data_dir, "train")
val_dir = os.path.join(output_data_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for category in os.listdir(input_data_dir):
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    category_path = os.path.join(input_data_dir,category)
    limit = int(len(os.listdir(category_path)) * 0.8)
    for j, image in enumerate(os.listdir(category_path)):
        source_file = os.path.join(input_data_dir, category, image)
        if j < limit:
            destination_folder = os.path.join(train_dir, category)
        else:
            destination_folder = os.path.join(val_dir, category)
        shutil.copy(source_file, destination_folder)