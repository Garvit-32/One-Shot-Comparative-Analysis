
import zipfile
import os
from skimage import io, transform, img_as_ubyte

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir


# Parameters
dataset_zip_files = ['images_background.zip', 'images_evaluation.zip']
raw_omniglot_location = DATA_PATH + '/Omniglot_Raw/'
prepared_omniglot_location = DATA_PATH + '/Omniglot/'
output_shape = (28, 28)


def handle_characters(alphabet_folder, character_folder, rotate):
    for root, _, character_images in os.walk(character_folder):
        character_name = root.split('/')[-1]
        mkdir(f'{alphabet_folder}.{rotate}/{character_name}')
        for img_path in character_images:
            # print(root+'/'+img_path)
            img = io.imread(root+'/'+img_path)
            img = transform.rotate(img, angle=rotate)
            img = transform.resize(img, output_shape, anti_aliasing=True)
            img = (img - img.min()) / (img.max() - img.min())

            io.imsave(
                f'{alphabet_folder}.{rotate}/{character_name}/{img_path}', img_as_ubyte(img))


def handle_alphabet(folder):
    print('{}...'.format(folder.split('/')[-1]))
    for rotate in [0, 90, 180, 270]:
        # Create new folders for each augmented alphabet
        mkdir(f'{folder}.{rotate}')
        for root, character_folders, _ in os.walk(folder):
            for character_folder in character_folders:

                handle_characters(folder, root + '/' +
                                  character_folder, rotate)

    rmdir(folder)


rmdir(prepared_omniglot_location)
mkdir(prepared_omniglot_location)


for root, _, files in os.walk(raw_omniglot_location):
    for f in files:
        if f in dataset_zip_files:
            print('Unzipping {}...'.format(f))
            zip_ref = zipfile.ZipFile(root + f, 'r')
            zip_ref.extractall(prepared_omniglot_location)
            zip_ref.close()

print('Processing background set...')
for root, alphabets, _ in os.walk(prepared_omniglot_location + 'images_background/'):
    for alphabet in sorted(alphabets):
        handle_alphabet(root + alphabet)

print('Processing evaluation set...')
for root, alphabets, _ in os.walk(prepared_omniglot_location + 'images_evaluation/'):
    for alphabet in sorted(alphabets):
        handle_alphabet(root + alphabet)
