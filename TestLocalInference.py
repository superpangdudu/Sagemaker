import torch
from transformers import pipeline
from diffusers.utils import load_image
import numpy as np
from PIL import Image
from controlnet_aux import LineartDetector

from transformers import BlipProcessor, BlipForConditionalGeneration

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDIMScheduler
)


#########################################################################################
schedulers = {
    'UniPCMultistepScheduler': UniPCMultistepScheduler,
    'EulerDiscreteScheduler': EulerDiscreteScheduler,
    'EulerAncestralDiscreteScheduler': EulerAncestralDiscreteScheduler,
    'HeunDiscreteScheduler': HeunDiscreteScheduler,
    'LMSDiscreteScheduler': LMSDiscreteScheduler,
    'KDPM2DiscreteScheduler': KDPM2DiscreteScheduler,
    'KDPM2AncestralDiscreteScheduler': KDPM2AncestralDiscreteScheduler,
    'DDIMScheduler': DDIMScheduler
}

#########################################################################################
model_config = {
    'base_model': {
        'name': 'stable-diffusion-v1-5',
        'path': 'runwayml/stable-diffusion-v1-5'
    },
    'controlnets': [
        {
            'name': 'tile',
            'path': 'lllyasviel/control_v11f1e_sd15_tile',
            'scale_from': 0.8,
            'scale_to': 1.1
        },
        {
            'name': 'lineart',
            'path': 'lllyasviel/control_v11p_sd15_lineart',
            'scale_from': 0.8,
            'scale_to': 1.0
        }
    ],
    'loras': [
        {
            'name': 'add_detail',
            'path': '',
            'scale_from': 0.8,
            'scale_to': 1.2
        },
        {
            'name': 'epi_noiseoffset2',
            'path': '',
            'scale_from': 0.8,
            'scale_to': 1.0
        }
    ]
}

schedulers_name = 'UniPCMultistepScheduler'

# input image
image_path = ''
image_width = 512
image_height = 512
# output directory
output_path = "./controlnet_out/"

#
generator = torch.manual_seed(0)

prompt = ''
negative_prompt = ''
steps = 20
num_images_per_prompt = 1
blip_enabled = False
strength = 0.75


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


#########################################################################################
from safetensors.torch import load_file


def __load_lora(pipeline, lora_path, lora_weight=0.5):
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
class ControlnetConfig:
    """
    ControlnetConfig
    """
    def __init__(self,model_path, model_name, s_from, s_to):
        self.name = model_name
        self.model_path = model_path
        self.scale_from = s_from
        self.scale_to = s_to

    def new_control_net_model(self):
        return ControlNetModel.from_pretrained(self.model_path, torch_dtype=torch.float16)

    def get_controlnet_image(self, img):
        return img

    def get_scale(self):
        return self.scale_from, self.scale_to


class LineartControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        from controlnet_aux import LineartDetector

        super().__init__(model_path, 'lineart', s_from, s_to)
        self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    def get_controlnet_image(self, img):
        out = self.processor(img)
        return out


class DepthControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        from transformers import pipeline as transformers_pipeline

        super().__init__(model_path, 'depth', s_from, s_to)
        self.depth_estimator = transformers_pipeline('depth-estimation')

    def get_controlnet_image(self, img):
        out = self.depth_estimator(img)['depth']
        return out


class TileControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        super().__init__(model_path, 'tile', s_from, s_to)

    def get_controlnet_image(self, img):
        return img
        #return resize_for_condition_image(img, 1024)


def get_controlnet(cfg_name, model_path, s_from, s_to):
    if cfg_name == 'lineart':
        return LineartControlnetConfig(model_path, s_from, s_to)
    elif cfg_name == 'depth':
        return DepthControlnetConfig(model_path, s_from, s_to)
    elif cfg_name == 'tile':
        return TileControlnetConfig(model_path, s_from, s_to)

    return None


#
inference_param = {}
inference_param['prompt'] = prompt
inference_param['negative_prompt'] = negative_prompt
inference_param['steps'] = steps
inference_param['num_images_per_prompt'] = num_images_per_prompt
inference_param['width'] = image_width
inference_param['height'] = image_height
inference_param['generator'] = generator
inference_param['strength'] = strength


#########################################################################################
def make_inference_scale_params(scale_list):
    result = []
    tmp = []
    for i in range(len(scale_list)):
        n, f, t = scale_list[i]

        is_empty_result = len(result) == 0

        current_scale = f
        while current_scale <= t:
            if is_empty_result:
                d = f'{n}_{current_scale}'
                result.append((d, [current_scale]))
            else:
                for x in range(len(result)):
                    d, l = result[x]
                    d += f'_{n}_{current_scale}'
                    tmp_list = l.copy()
                    tmp_list.append(current_scale)
                    tmp.append((d, tmp_list))

            current_scale += 0.1
            current_scale = round(current_scale, 1)

        if not is_empty_result:
            result = tmp
            tmp = []

    return result


#
controlnet_configs = []
controlnet_scale_list = []
for i in range(len(model_config['controlnets'])):
    name = model_config['controlnets'][i]['name']
    path = model_config['controlnets'][i]['path']
    scale_from = model_config['controlnets'][i]['scale_from']
    scale_to = model_config['controlnets'][i]['scale_to']

    #
    controlnet_scale_list.append((name, scale_from, scale_to))

    # FIXME
    #cfg = get_controlnet(name, path, scale_from, scale_to)
    #controlnet_configs.append(cfg)


controlnet_scale_params = make_inference_scale_params(controlnet_scale_list)

#
loras = []
lora_scale_list = []
for i in range(len(model_config['loras'])):
    name = model_config['loras'][i]['name']
    path = model_config['loras'][i]['path']
    scale_from = model_config['loras'][i]['scale_from']
    scale_to = model_config['loras'][i]['scale_to']

    loras.append((name, path))
    lora_scale_list.append((name, scale_from, scale_to))

lora_scale_params = make_inference_scale_params(lora_scale_list)


#########################################################################################
def get_controlnet_pipeline(model, controlnet):
    controlnet_pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model,
                                                                                   controlnet=controlnet,
                                                                                   torch_dtype=torch.float16,
                                                                                   safety_checker=None)
    controlnet_pipeline.to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    controlnet_pipeline.scheduler = schedulers[schedulers_name].from_config(controlnet_pipeline.scheduler.config)
    controlnet_pipeline.enable_model_cpu_offload()
    controlnet_pipeline.enable_xformers_memory_efficient_attention()

    return controlnet_pipeline


#
image = load_image(image_path)
image = image.resize((image_width, image_height))

controlnets = []
controlnet_images = []
for i in range(len(controlnet_configs)):
    cfg = controlnet_configs[i]
    cn = cfg.new_control_net_model()
    controlnets.append(cn)

    cn_img = cfg.get_controlnet_image(image)
    controlnet_images.append(image)

inference_param['image'] = image
inference_param['control_image'] = controlnet_images

#
if blip_enabled:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    blip_input = blip_processor(image, return_tensors="pt").to("cuda")
    blip_out = blip_model.generate(**blip_input)
    blip_prompt = blip_processor.decode(blip_out[0], skip_special_tokens=True)

    print(blip_prompt)
    prompt = blip_prompt + ', ' + prompt

#
base_mode_name = model_config['base_model']['name']
base_mode_path = model_config['base_model']['path']

for i in range(len(lora_scale_params)):
    d, lora_scales = lora_scale_params[i]

    controlnet_pipeline = get_controlnet_pipeline(base_mode_path, controlnets)
    # load loras
    for x in range(len(lora_scales)):
        lora_name, lora_path = loras[x]
        controlnet_pipeline = __load_lora(controlnet_pipeline,
                                          lora_path,
                                          lora_scales[x])
    # inference for controlnet scale params
    for y in range(len(controlnet_scale_params)):
        controlnet_description, controlnet_scales = controlnet_scale_params
        inference_param['controlnet_conditioning_scale'] = controlnet_scales

        image_name_prefix = f'{controlnet_description}_{d}'

        output_images = controlnet_pipeline(
            **inference_param
        ).images

        for image_count in range(len(output_images)):
            image_name = f'{output_path}{image_name_prefix}_{image_count}.jpg'
            output_images[image_count].save(image_name)
