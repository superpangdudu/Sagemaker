
import argparse
import os

from tqdm.auto import tqdm


#########################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Tool used to create metadata.jsonl.")

    parser.add_argument(
        '--output_dir', type=str, default='', required=False,
        help='Output directory..'
    )

    parser.add_argument(
        '--image_caption_dir', type=str, default='', required=False,
        help='Images and captions directory.'
    )

    #
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def get_all_files(dir):
    files = []
    return os.listdir(dir)


def read_file(path):
    with open(path, encoding='utf-8') as caption_file:
        content = caption_file.read()
        return content


#########################################################################################
if __name__ == '__main__':
    args = parse_args()

    # ensure directory exists
    # if os.path.exists(args.output_dir) is not True:
    #     os.mkdir(args.output_dir)

    args.image_caption_dir = r'C:\Users\Administrator\Desktop\dudu-img\all\preprocessed'

    #
    files = os.listdir(args.image_caption_dir)
    image_captions = {}

    for f in files:
        if f.endswith('.png') is not True \
            and f.endswith('.jpg') is not True:
            continue

        image = f
        prefix = image[0:-4]

        caption = prefix + '.txt'
        n = files.index(caption)
        if caption not in files:
            continue

        image_captions[prefix] = {'image': image, 'caption': caption}

    #
    output_file = args.output_dir + '/' + 'metadata.jsonl'
    output_file = 'metadata.jsonl'
    with open(output_file, 'w+') as out:
        for k in image_captions.keys():
            image = image_captions[k]['image']
            caption = args.image_caption_dir + '/' + image_captions[k]['caption']

            caption = read_file(caption)

            line = f'{{""file_name": "{image}", "text": "{caption}"}}\n'
            out.write(line)






