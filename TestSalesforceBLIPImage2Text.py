
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_path = "/home/zlkh000/krly/lifei/ai-models/Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path,
                                                     torch_dtype=torch.float16).to("cuda")

image_path = "/home/zlkh000/krly/lifei/test-images/dudu-all/2.png"
raw_image = Image.open(image_path).convert('RGB')

#text = "a photography of"
#inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))