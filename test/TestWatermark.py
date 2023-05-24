
from PIL import Image

image = Image.open('e:/a.jpg').convert("RGB")
watermark = Image.open('z:/download/watermark.png').convert("RGBA")

image_width, image_height = image.size
watermark_width, watermark_height = watermark.size

layer = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
layer.paste(watermark, (image_width - (watermark_width + 10), image_height - (watermark_height + 10)))

out = Image.composite(layer, image, layer)
out.show()
# out.save('e:/hahahahahaha.jpg')
