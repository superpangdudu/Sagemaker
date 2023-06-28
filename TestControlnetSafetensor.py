
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch


model_path = 'E:/test/Stable-Diffusion/diffusers/scripts/tmp/t2iadapter_color-fp16.safetensors'

controlnet = ControlNetModel.from_pretrained(
    'webui/ControlNet-modules-safetensors',
    torch_dtype=torch.float16,
    use_safetensors=True
)

print(controlnet)