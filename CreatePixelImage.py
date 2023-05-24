
import argparse

import numpy as np
from PIL import Image, ImageTk

import os
import random

from tqdm.auto import tqdm


#########################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Tool used to create random pixel images.")

    parser.add_argument(
        '--image_width', type=int, default=512, required=False,
        help='Width of the image.'
    )

    parser.add_argument(
        '--image_height', type=int, default=512, required=False,
        help='Height of the image.'
    )

    parser.add_argument(
        '--pixel_row', type=int, default=64, required=False,
        help='Rows in the image.'
    )

    parser.add_argument(
        '--pixel_column', type=int, default=64, required=False,
        help='Columns in the image.'
    )

    parser.add_argument(
        '--count', type=int, default=2000, required=False,
        help='Count of output pixel images in the image.'
    )

    parser.add_argument(
        '--output_dir', type=str, default='PixelImages', required=False,
        help='Path to the output directory.'
    )

    #
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def create_pixel_image(path, img_width, img_height, pixel_row, pixel_column):
    # RGB
    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    pixel_grid_width = int(img_width / pixel_column)
    pixel_grid_height = int(img_height / pixel_row)

    for row in range(pixel_row):
        for col in range(pixel_column):
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)

            width_offset = col * pixel_grid_width
            height_offset = row * pixel_grid_height

            # canvas[height_offset: height_offset + pixel_grid_height,
            #         width_offset: width_offset + pixel_grid_width][0] = red
            # canvas[height_offset: height_offset + pixel_grid_height,
            #         width_offset: width_offset + pixel_grid_width][1] = green
            # canvas[height_offset: height_offset + pixel_grid_height,
            #         width_offset: width_offset + pixel_grid_width][2] = blue

            for pr in range(pixel_grid_height):
                for pc in range(pixel_grid_width):
                    r = pr + height_offset
                    c = pc + width_offset

                    canvas[c][r][0] = red
                    canvas[c][r][1] = green
                    canvas[c][r][2] = blue


    # for r in range(img_width):
    #     for c in range(img_height):
    #         red = random.randint(0, 255)
    #         green = random.randint(0, 255)
    #         blue = random.randint(0, 255)
    #
    #         canvas[c][r][0] = red
    #         canvas[c][r][1] = green
    #         canvas[c][r][2] = blue

    #
    image = Image.fromarray(canvas)
    image.resize()
    image.save(path)


if __name__ == '__main__':
    # xxx = Image.open('e:/a.jpg')
    # xxx = xxx.resize((64, 64), resample=Image.BICUBIC)
    # xxx.save('e:/axf.jpg')

    args = parse_args()

    # ensure directory exists
    if os.path.exists(args.output_dir) is not True:
        os.mkdir(args.output_dir)

    #
    bar = tqdm(total=args.count)
    bar.set_description_str('Processing')

    #
    for i in range(args.count):
        image_path = '%s/%d.png' %(args.output_dir, i)
        create_pixel_image(image_path, args.image_width, args.image_height, args.pixel_row, args.pixel_column)
        bar.update(1)
