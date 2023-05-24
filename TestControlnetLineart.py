import torch
from transformers import pipeline
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import LineartDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

#########################################################################################
from safetensors.torch import load_file


def __load_lora(
        pipeline
        , lora_path
        , lora_weight=0.5
):
    state_dict = load_file(lora_path)
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    alpha = lora_weight
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue

        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER + '_')[-1].split('_')
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET + '_')[-1].split('_')
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_' + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

            # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline



#########################################################################################
image_path = "/home/zlkh000/krly/lifei/test-images/dudu-all/17.png"
#image_path = "https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png"
prompt = "realistic, 1girl, masterpiece, pencil drawing, portrait, sketch, painting, rough sketch, line art, meticulous painting, white paper, character on paper, black and white, extra lines,clear lines, shadow"
steps = 20
num_images_per_prompt = 4
controlnet_conditioning_scale = 1.1

negative_prompt = 'ugly, distort, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, poorly drawn face, poorly drawn eyes, poorly drawn nose, poorly drawn mouth, poorly Rendered face, disfigured, deformed body features, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

#########################################################################################
controlnet_line_art_checkpoint = "/home/zlkh000/krly/lifei/ai-models/lllyasviel/control_v11p_sd15_lineart"
controlnet_depth_checkpoint = "/home/zlkh000/krly/lifei/ai-models/lllyasviel/sd-controlnet-depth"

sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt"
sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/basilKorea2dfilter_v10"
sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/handPaintedPortrait_sdv15_lora"
sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/handPaintedPortrait_basilKorea2dfilter_lora"

sd_checkpoints = [
    ("/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt", "sd15"),
    ("/home/zlkh000/krly/lifei/ai-models/handPaintedPortrait_sdv15_lora", "sd15-handPaintedPortrait"),
    ("/home/zlkh000/krly/lifei/ai-models/handPaintedPortrait_basilKorea2dfilter_lora", "basilKorea2dfilter-handPaintedPortrait")
]

#
image = load_image(image_path)
image = image.resize((512, 512))

#processor = LineartDetector.from_pretrained("/home/zlkh000/krly/lifei/ai-models/lllyasviel/Annotators")
processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

lineart_image = processor(image)
lineart_image.save("./controlnet_out/lineart_image.png")

controlnet_lineart = ControlNetModel.from_pretrained(controlnet_line_art_checkpoint, torch_dtype=torch.float16)

#########################################################################################
depth_estimator = pipeline('depth-estimation')

depth_image = depth_estimator(image)['depth']
depth_image.save("./controlnet_out/depth_image.png")
#depth_image = load_image('/home/zlkh000/krly/lifei/test/depth.png')

controlnet_depth = ControlNetModel.from_pretrained(controlnet_depth_checkpoint, torch_dtype=torch.float16)

#########################################################################################
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_checkpoint,
    controlnet=[controlnet_depth, controlnet_lineart],
    torch_dtype=torch.float16
)

#pipe.unet.load_attn_procs('/home/zlkh000/krly/lifei/ai-models/handPaintedPortrait_lora')
#pipe.to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

generator = torch.manual_seed(0)
images = pipe(prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            image=[depth_image, lineart_image]).images
for i in range(num_images_per_prompt):
    image = images[i]
    image.save('out/image_out' + str(i) + '.png')

#########################################################################################
# for i in range(len(sd_checkpoints)):
#     ckpt, name = sd_checkpoints[i]
#     pipe = StableDiffusionControlNetPipeline.from_pretrained(
#         ckpt,
#         controlnet=[controlnet_depth, controlnet_lineart],
#         torch_dtype=torch.float16
#     )
#     pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.enable_model_cpu_offload()
#     pipe.enable_xformers_memory_efficient_attention()
#
#     generator = torch.manual_seed(0)
#     images = pipe(prompt,
#                   negative_prompt=negative_prompt,
#                   num_inference_steps=steps,
#                   num_images_per_prompt=num_images_per_prompt,
#                   controlnet_conditioning_scale=controlnet_conditioning_scale,
#                   generator=generator,
#                   image=[depth_image, lineart_image]).images
#     for x in range(num_images_per_prompt):
#         image = images[x]
#         image.save('controlnet_out/image_out_' + name + '_' + str(x) + '.png')





























