from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import argparse
import os

from tqdm.auto import tqdm


#########################################################################################
model_path = '/home/zlkh000/krly/lifei/ai-models/Salesforce/blip-image-captioning-base'

#########################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Tool used to create image captions.")

    parser.add_argument(
        '--output_dir', type=str, default='.', required=False,
        help='Output directory..'
    )

    parser.add_argument(
        '--images_dir', type=str, default='.', required=False,
        help='Images directory.'
    )

    #
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def get_all_images_in_dir(image_dir):
    files = os.listdir(image_dir)
    imgs = []

    for f in files:
        f = f.lower()
        if f.endswith('bmp') \
            or f.endswith('jpg') \
            or f.endswith('jpeg') \
            or f.endswith('png'):
            imgs.append(f)
    return imgs



#########################################################################################
if __name__ == '__main__':
    args = parse_args()

    print("Starting......")

    #
    images = get_all_images_in_dir(args.images_dir)

    #
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to('cuda')

    print('Model loading done')

    #
    bar = tqdm(total=len(images))
    bar.set_description_str('Processing')

    #
    for img_file_path in images:
        img = Image.open(img_file_path).convert('RGB')
        inputs = processor(img, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        out_file_path = os.path.basename(img_file_path)
        out_file_path, e = os.path.splitext(out_file_path)
        out_file_path = args.output_dir + '/' + out_file_path + '.txt'

        with open(out_file_path, 'w+') as out:
            out.write(caption)

        #
        bar.update(1)
