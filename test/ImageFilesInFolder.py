

import os
import PIL.Image as Image


# get all sub folders in the given folder
def get_sub_folders(folder):
    folders = []
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders


# get files in the given folder
def get_files(folder):
    results = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            results.append(os.path.join(root, f))
    return results


# get image size
def get_image_size(file):
    img = Image.open(file)
    return img.size


directories = get_sub_folders('E:/test/android/SPOTLED/app/src/main/res')
files = []

for directory in directories:
    files += get_files(directory)

for file in files:
    if file.endswith('.png') or file.endswith('.jpg'):
        size = get_image_size(file)

        file = file.replace('E:/test/android/', '')

        print(f'{file}, {size}')


