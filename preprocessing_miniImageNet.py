from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir


# Clean up folders
rmdir(DATA_PATH + '/miniImageNet/images_background')
rmdir(DATA_PATH + '/miniImageNet/images_evaluation')
mkdir(DATA_PATH + '/miniImageNet/images_background')
mkdir(DATA_PATH + '/miniImageNet/images_evaluation')

# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))

# Train/test split
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:80], classes[80:]

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_background/{c}/')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_evaluation/{c}/')

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images'):
    for f in tqdm(files, total=600*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = f'{root}/{f}'
            dst = DATA_PATH + \
                f'/miniImageNet/{subset_folder}/{class_name}/{image_name}'
            shutil.copy(src, dst)
