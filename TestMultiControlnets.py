import torch
from transformers import pipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
from controlnet_aux import LineartDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler
)


schedulers = {
    'UniPCMultistepScheduler': UniPCMultistepScheduler,
    'EulerDiscreteScheduler' : EulerDiscreteScheduler,
    'EulerAncestralDiscreteScheduler' : EulerAncestralDiscreteScheduler,
    'HeunDiscreteScheduler' : HeunDiscreteScheduler,
    'LMSDiscreteScheduler' : LMSDiscreteScheduler,
    'KDPM2DiscreteScheduler' : KDPM2DiscreteScheduler,
    'KDPM2AncestralDiscreteScheduler' : KDPM2AncestralDiscreteScheduler,
    'DDIMScheduler': DDIMScheduler
}

schedulers_name = 'UniPCMultistepScheduler'

# lora scale start
lora_init_scale = 0.3
# lora scale end
lora_max_scale = 1.0

#
guidance_init_scale = 5.0
guidance_max_scale = 15.0

# controlnet scale start
controlnet_init_scale = 0.8
# controlnet scale end
controlnet_max_scale = 1.2

# input image
image_path = "/home/zlkh000/krly/lifei/test-images/dudu-all/2.png"
image_width = 512
image_height = 512
# output directory
output_path = "./controlnet_out/"

#
prompt = "arafed girl standing in front of a fountain in a park, realistic, pencil drawing, portrait, sketch, painting, rough sketch, line art, meticulous painting, white paper, character on paper, black and white, extra lines,clear lines, shadow"
#prompt = "arafed girl standing in front of a fountain in a park"
negative_prompt = 'ugly, distort, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, poorly drawn face, poorly drawn eyes, poorly drawn nose, poorly drawn mouth, poorly Rendered face, disfigured, deformed body features, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

#
steps = 20
num_images_per_prompt = 1

#
sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt"
sd_checkpoint = "/home/zlkh000/krly/lifei/ai-models/basilKorea2dfilter_v10"

#
controlnet_line_art_checkpoint = "/home/zlkh000/krly/lifei/ai-models/lllyasviel/control_v11p_sd15_lineart"
controlnet_depth_checkpoint = "/home/zlkh000/krly/lifei/ai-models/lllyasviel/sd-controlnet-depth"
# controlnet_line_art_checkpoint = "lllyasviel/control_v11p_sd15_lineart"
# controlnet_depth_checkpoint = "lllyasviel/control_v11p_sd15_depth"
controlnet_tile_checkpoint = "lllyasviel/control_v11f1e_sd15_tile"

# lora weights path
lora = "/home/zlkh000/krly/lifei/ai-models/Sketch/handPaintedPortrait_v12.safetensors"
lora_scale = lora_init_scale

#
generator = torch.manual_seed(0)

#########################################################################################
image = load_image(image_path)
image = image.resize((image_width, image_height))

#processor = LineartDetector.from_pretrained("/home/zlkh000/krly/lifei/ai-models/lllyasviel/Annotators")
processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

lineart_image = processor(image)
lineart_image.save(output_path + "lineart_image.png")

controlnet_lineart = ControlNetModel.from_pretrained(controlnet_line_art_checkpoint, torch_dtype=torch.float16)

#########################################################################################
depth_estimator = pipeline('depth-estimation')

depth_image = depth_estimator(image)['depth']
depth_image.save(output_path + "depth_image.png")

controlnet_depth = ControlNetModel.from_pretrained(controlnet_depth_checkpoint, torch_dtype=torch.float16)

#########################################################################################
def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

tile_image = resize_for_condition_image(image, 1024)

controlnet_tile = ControlNetModel.from_pretrained(controlnet_tile_checkpoint, torch_dtype=torch.float16)

#########################################################################################
#
controlnets_config = [
    (controlnet_lineart, controlnet_init_scale, controlnet_max_scale, "lineart", lineart_image),
    #(controlnet_depth, controlnet_init_scale, controlnet_max_scale, "depth", depth_image),
    (controlnet_tile, controlnet_init_scale, controlnet_max_scale, "tile", tile_image)
]

controlnets = []
images = []
for i in range(len(controlnets_config)):
    c, _, _, _, img = controlnets_config[i]
    controlnets.append(c)
    images.append(img)

controlnet_params = []
tmp = []

def getControlnetParams(idx=0):
    if idx >= len(controlnets_config):
        return

    c, start, end, name, _ = controlnets_config[idx]
    a = np.arange(start, end + 0.1, 0.1)
    a = np.around(a, decimals=1)

    if idx == 0:
        for i in a:
            tmp.append((name + '_' + str(i), [i]))
            getControlnetParams(idx + 1)
    else:
        n, l = tmp.pop()
        for i in a:
            x = n + '_' + name + '_' + str(i)
            y = l.copy()
            y.append(i)

            if idx == len(controlnets_config) - 1:
                controlnet_params.append((x, y))
            else:
                tmp.append((x, y))

getControlnetParams()

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
def getControlnetPipeline():
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_checkpoint,
            controlnet=controlnets,
            torch_dtype=torch.float16
        )
    pipe.to("cuda")
    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = schedulers[schedulers_name].from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

#
while len(controlnet_params) > 0:
    controlnet_name, controlnet_scaleList = controlnet_params.pop()
    while lora_scale <= lora_max_scale:
        pipe = getControlnetPipeline()
        pipe_lora = __load_lora(pipeline=pipe, lora_path=lora, lora_weight=lora_scale)

        # for guidance scale
        guidance_scale = guidance_init_scale
        while guidance_scale <= guidance_max_scale:
            outputs = pipe_lora(prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=steps,
                                num_images_per_prompt=num_images_per_prompt,
                                controlnet_conditioning_scale=controlnet_scaleList,
                                generator=generator,
                                width=image_width,
                                height=image_height,
                                guidance_scale=guidance_scale,
                                image=images).images
            for i in range(num_images_per_prompt):
                image = outputs[i]
                name = f'{controlnet_name}_lora_{lora_scale}_guidance_{guidance_scale}_{str(i)}.png'
                image.save(output_path + name)

            guidance_scale += 0.1
            guidance_scale = round(guidance_scale, 1)

        #
        guidance_scale = guidance_init_scale

        #
        lora_scale += 0.1
        lora_scale = round(lora_scale, 1)
    lora_scale = lora_init_scale
































