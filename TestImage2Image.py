
import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

device = 'cuda'
#device = 'cpu'

#pretrained_model_path = 'E:/test/ai-model/stable-diffusion/stable-diffusion-v1-5'
pretrained_model_path = '/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt'
lora_path = "/home/zlkh000/krly/lifei/ai-models/sd-dudu-head-model-lora"

#pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_path)
pipe.unet.load_attn_procs(lora_path)
pipe = pipe.to(device)

#url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
# url = 'http://c-ssl.duitang.com/uploads/item/201410/29/20141029124807_ySRA3.jpeg'
#
# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))
# init_image.show()
imgPath = '1.png'
init_image = Image.open(imgPath).convert('RGB')
init_image = init_image.resize((512, 512))

prompt = "A fantasy landscape, trending on art station"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save('xxx.png')
