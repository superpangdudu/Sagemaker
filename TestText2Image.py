

import torch
from diffusers import StableDiffusionPipeline


pretrained_model_path = 'E:/test/ai-model/stable-diffusion/stable-diffusion-v1-5'

#pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
#pipe = pipe.to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_path)
pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.show()
