from PIL import Image, ImageDraw, ImageFont

width = 1920
height = 1024

output_dir = 'z:/download/images/'

#
font = ImageFont.truetype('arial.ttf', size=500)

for i in range(10):
    text = str(i + 1)

    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = draw.textbbox((0, 0), text=text, font=font)
    text_width = right - left
    text_height = bottom - top

    #text_width, text_height = draw.textsize(text, font=font)
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 2

    draw.text((x, y), text, fill='black', font=font)
    image.save(f'{output_dir}{i+1}.jpg')

