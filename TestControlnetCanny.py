

import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image


image_path = '/home/zlkh000/krly/lifei/test-images/dudu-all/17.png'

image = load_image(image_path)
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.unet.load_attn_procs('/home/zlkh000/krly/lifei/ai-models/sd-dudu-head-model-lora')
pipe.to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

#
negative_prompt = 'ugly, distort, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, poorly drawn face, poorly drawn eyes, poorly drawn nose, poorly drawn mouth, poorly Rendered face, disfigured, deformed body features, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

prompt = 'a dudu girl in iron man suite'

count = 50
num_images_per_prompt = 4
for i in range(count):
    images = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                 image=image,
                 num_images_per_prompt=num_images_per_prompt,
                 num_inference_steps=100).images
    for x in range(num_images_per_prompt):
        image = images[x]
        image.save('out/cn' + str(i) + '-' + str(x) + '.png')

