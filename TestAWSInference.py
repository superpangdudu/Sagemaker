# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import uuid
import io
import sys
import tarfile
import traceback
import uuid

import logging

from PIL import Image

import requests
import boto3
import sagemaker
import torch
import s3fs

from collections import defaultdict
from torch import autocast
from diffusers import StableDiffusionPipeline,\
    StableDiffusionImg2ImgPipeline,\
    StableDiffusionControlNetPipeline,\
    ControlNetModel
from diffusers import AltDiffusionPipeline, AltDiffusionImg2ImgPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, \
    LMSDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DDIMScheduler

from safetensors.torch import load_file, save_file

#
logging.basicConfig(level=logging.INFO)

#########################################################################################
s3_client = boto3.client('s3')

max_height = os.environ.get("max_height", 768)
max_width = os.environ.get("max_width", 768)
max_steps = os.environ.get("max_steps", 100)
max_count = os.environ.get("max_count", 4)
s3_bucket = os.environ.get("s3_bucket", "")
custom_region = os.environ.get("custom_region", None)
safety_checker_enable = json.loads(os.environ.get("safety_checker_enable", "false"))

# add lora support
lora_model = os.environ.get("lora_model", None)

# need add more sampler
samplers = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "eular": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "ddim": DDIMScheduler
}

scheduler = 'dpm2_a'
#########################################################################################
'''
Model path
Base model: 'models/base/{model name}'
Controlnet: 'models/controlnet/{controlnet model name}'
Lora: 'models/lora/{LoRA model name}'
'''

model_config = {
    'base_model': {
        'name': 'stable-diffusion-v1-5',
        'path': 's3://model/base/stable-diffusion-v1-5.tar.gz',
        'default': 'runwayml/stable-diffusion-v1-5'
    },
    'controlnets' : [
        {
            'name' : 'tile',
            'path' : 's3://model/controlnet/control_v11f1e_sd15_tile',
            'default' : 'lllyasviel/control_v11f1e_sd15_tile',
            'scale' : 0.8
        },
        {
            'name' : 'lineart',
            'path' : 's3://model/controlnet/control_v11p_sd15_lineart',
            'default' : 'lllyasviel/control_v11p_sd15_lineart',
            'scale' : 0.8
        }
    ],
    'loras' : [
        {
            'name' : 'handPaintedPortrait_v12',
            'path' : 's3://models/lora/handPaintedPortrait_v12.safetensors',
            'scale' : 0.8
        }
    ]
}

#########################################################################################
class ControlnetConfig:
    def __init__(self, remote_model_path, model_parent_dir, model_name, scale):
        self.remote_model_path = remote_model_path
        self.parent_dir = model_parent_dir
        self.name = model_name
        self.model_path = model_parent_dir + model_name
        self.scale = scale

    def newInstance(self):
        return ControlNetModel.from_pretrained(self.model_path, torch_dtype=torch.float16)

    def getControlnetImage(self, img):
        return img

    def getScale(self):
        return self.scale

class LineartControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, model_parent_dir, scale):
        from controlnet_aux import LineartDetector

        super().__init__(remote_model_path, model_parent_dir, 'lineart', scale)
        self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        logging.info('######################  LineartControlnetConfig.processor done')

    def getControlnetImage(self, img):
        return self.processor(img)

class DepthControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, model_parent_dir, scale):
        from transformers import pipeline as transformers_pipeline

        super().__init__(remote_model_path, model_parent_dir, 'depth', scale)
        self.depth_estimator = transformers_pipeline('depth-estimation')
        logging.info('######################  DepthControlnetConfig.depth_estimator done')

    def getControlnetImage(self, img):
        return self.depth_estimator(img)['depth']

class TileControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, model_parent_dir, scale):
        super().__init__(remote_model_path, model_parent_dir, 'tile', scale)

def getControlnet(remote_model_path, model_parent_dir, name, scale):
    if name == 'lineart':
        return LineartControlnetConfig(remote_model_path, model_parent_dir, scale)
    elif name == 'depth':
        return DepthControlnetConfig(remote_model_path, model_parent_dir, scale)
    elif name == 'tile':
        return TileControlnetConfig(remote_model_path, model_parent_dir, scale)

    return None

#########################################################################################
#
MODEL_ROOT_PATH = '/model/'

#
BASE_MODEL_NAME = model_config['base_model']['name']
BASE_MODEL_REMOTE_PATH = model_config['base_model']['path']
base_model_path = MODEL_ROOT_PATH + 'base/' + BASE_MODEL_NAME

logging.info('#' * 90)
logging.info('Base model name: %s', BASE_MODEL_NAME)
logging.info('Base model remote path: %s', BASE_MODEL_REMOTE_PATH)
logging.info('Base model local path: %s', base_model_path)

#
CONTROLNET_ROOT_PATH = MODEL_ROOT_PATH + 'controlnet/'
controlnet_configs = []
for i in range(len(model_config['controlnets'])):
    name = model_config['controlnets'][i]['name']
    remote_path = model_config['controlnets'][i]['path']
    scale = model_config['controlnets'][i]['scale']

    controlnetConfig = getControlnet(remote_path,
                                     CONTROLNET_ROOT_PATH,
                                     name,
                                     scale)
    controlnet_configs.append(controlnetConfig)

    logging.info('#' * 90)
    logging.info('Controlnet config: %s, scale: %f, remote path: %s', name, scale, remote_path)

#
LORA_MODELS_NAME = []
LORA_MODELS_REMOTE_PATH = []
LORA_MODELS_SCALE = []

LORA_ROOT_PATH = MODEL_ROOT_PATH + 'lora/'
loras = model_config['loras']
for i in range(len(loras)):
    LORA_MODELS_NAME.append(loras[i]['name'])
    LORA_MODELS_REMOTE_PATH.append(loras[i]['path'])
    LORA_MODELS_SCALE.append(loras[i]['scale'])

    logging.info('#' * 90)
    logging.info('LoRA: %s, scale: %f, remote path: %s', loras[i]['name'], loras[i]['scale'], loras[i]['path'])

#
IMG2IMG_FLAG = True

#
def download_from_s3(remote, local):
    fs = s3fs.S3FileSystem()
    fs.get(remote, local, recursive=True)

    # s3 = boto3.resource('s3')
    # s3.meta.client.download_file('sagemaker-us-east-2-290106689812',
    #                              'model/lora/handPaintedPortrait_v12.safetensors',
    #                              '/tmp/handPaintedPortrait_v12.safetensors')

    # fs = s3fs.S3FileSystem()
    # fs.get('s3://sagemaker-us-east-2-290106689812/model/lora/', '/tmp/lora', recursive=True)

def untar_from_s3_to_local(remote, local):
    """
    Download tar.gz file from S3, and untar it to local path
    :param remote: tar.gz in S3
    :param local: directory to untar
    :return: None
    """
    f = local + '/model.tar.gz'
    os.makedirs(local)
    download_from_s3(remote, f)
    logging.info('File downloaded - from [%s] to [%s]', remote, f)

    t = tarfile.open(f)
    t.extractall(path=local)
    logging.info('Untar done - [%s]', local)

def prepare_models():
    # base model
    if not os.path.isdir(base_model_path):
        logging.info('Base model is not ready, trying to download......')
        os.makedirs(base_model_path)
        untar_from_s3_to_local(BASE_MODEL_REMOTE_PATH, base_model_path)
    logging.info('Base model weights ready, name - %s, path - %s', BASE_MODEL_NAME, base_model_path)

    # controlnets
    # FIXME
    for i in range(len(controlnet_configs)):
        controlnet_model_path = CONTROLNET_ROOT_PATH + controlnet_configs[i].name
        if not os.path.isdir(controlnet_model_path):
            os.makedirs(controlnet_model_path)
            download_from_s3(controlnet_configs[i].remote_model_path, controlnet_model_path)
            logging.info(f'Download controlnet {controlnet_model_path}')
        logging.info(f'controlnet {controlnet_model_path} is ready')

    # loras
    # FIXME
    for i in range(len(LORA_MODELS_NAME)):
        lora_model_path = LORA_ROOT_PATH + LORA_MODELS_NAME[i] + '.safetensors'
        if os.path.exists(lora_model_path) is not True:
            os.makedirs(LORA_ROOT_PATH)
            download_from_s3(LORA_MODELS_REMOTE_PATH[i], lora_model_path)
            logging.info(f'Download LoRA {lora_model_path}')
        logging.info(f'LoRA {lora_model_path} is ready')

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

prepare_models()
def model_fn(model_dir):
    #prepare_models()

    pipeline = None

    # for controlnets
    if len(controlnet_configs) == 0:
        if IMG2IMG_FLAG:
            pipeline = StableDiffusionImg2ImgPipeline(base_model_path,
                                                      torch_dtype=torch.float16)
        else:
            pipeline = StableDiffusionPipeline(base_model_path,
                                                torch_dtype=torch.float16)
    else:
        controlnets = []
        for i in range(len(controlnet_configs)):
            cn = ControlNetModel.from_pretrained(CONTROLNET_ROOT_PATH + controlnet_configs[i].name,
                                                 torch_dtype=torch.float16)
            controlnets.append(cn)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(base_model_path,
                                                                     controlnet=controlnets,
                                                                     torch_dtype=torch.float16)

    # controlnets = []
    # lineart_controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_lineart',
    #                                                      torch_dtype=torch.float16)
    # print('lineart_controlnet done')
    # tile_controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',
    #                                                      torch_dtype=torch.float16)
    # print('tile_controlnet done')
    #
    # controlnets.append(lineart_controlnet)
    # controlnets.append(tile_controlnet)
    #
    # pipeline = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
    #                                                              controlnet=controlnets,
    #                                                              torch_dtype=torch.float16)

    if pipeline is None:
        return None

    # FIXME
    for i in range(len(LORA_MODELS_NAME)):
        model = f"{LORA_ROOT_PATH}{LORA_MODELS_NAME[i]}.safetensors"
        pipeline = __load_lora(pipeline, model, LORA_MODELS_SCALE[i])

    #
    #pipeline.scheduler = samplers[scheduler].from_config(pipeline.scheduler.config)

    # TODO change scheduler
    pipeline.to('cuda')
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_attention_slicing()

    print('pipeline done')
    return pipeline

#########################################################################################
def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    # {
    # "prompt": "a photo of an astronaut riding a horse on mars",
    # "negative_prompt":"",
    # "steps":0,
    # "sampler":"",
    # "seed":-1,
    # "height": 512,
    # "width": 512
    # }
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    return prepare_opt(input_data)


def clamp_input(input_data, minn, maxn):
    """
    clamp_input check input
    """
    return max(min(maxn, input_data), minn)


def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key


def prepare_opt(input_data):
    """
    Prepare inference input parameter
    """
    opt = {}
    opt["prompt"] = input_data.get(
        "prompt", "a photo of an astronaut riding a horse on mars")
    opt["negative_prompt"] = input_data.get("negative_prompt", "")
    opt["steps"] = clamp_input(input_data.get(
        "steps", 20), minn=20, maxn=max_steps)
    opt["sampler"] = input_data.get("sampler", None)
    opt["height"] = clamp_input(input_data.get(
        "height", 512), minn=64, maxn=max_height)
    opt["width"] = clamp_input(input_data.get(
        "width", 512), minn=64, maxn=max_width)
    opt["count"] = clamp_input(input_data.get(
        "count", 1), minn=1, maxn=max_count)
    opt["seed"] = input_data.get("seed", 1024)
    opt["input_image"] = input_data.get("input_image", None)

    if opt["sampler"] is not None:
        opt["sampler"] = samplers[opt["sampler"]
        ] if opt["sampler"] in samplers else samplers["euler_a"]

    print(f"=================prepare_opt=================\n{opt}")
    return opt

def getMapItem(map, item, default):
    if item in map:
        return map[item]
    else:
        return default

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    print("=================predict_fn=================")
    print('input_data: ', input_data)
    prediction = []

    try:
        sagemaker_session = sagemaker.Session() if custom_region is None else sagemaker.Session(
            boto3.Session(region_name=custom_region))
        bucket = sagemaker_session.default_bucket()
        if s3_bucket != "":
            bucket = s3_bucket
        default_output_s3uri = f's3://{bucket}/stablediffusion/asyncinvoke/images/'
        output_s3uri = input_data['output_s3uri'] if 'output_s3uri' in input_data else default_output_s3uri
        infer_args = input_data['infer_args'] if (
                'infer_args' in input_data) else None
        print('infer_args: ', infer_args)
        init_image = infer_args['init_image'] if infer_args is not None and 'init_image' in infer_args else None
        input_image = input_data['input_image']
        print('init_image: ', init_image)
        print('input_image: ', input_image)

        # TODO download initial image from S3
        if input_image is not None:
            response = requests.get(input_image, timeout=5)
            init_img = Image.open(io.BytesIO(response.content)).convert("RGB")
            init_img = init_img.resize((input_data["width"], input_data["height"]))
            print('init_img download done')

        # prepare controlnet images
        controlnet_images = []
        for i in range(len(controlnet_configs)):
            img = controlnet_configs[i].getControlnetImage(init_image)
            controlnet_images.append(img)

        prompt = getMapItem(input_data, 'prompt', 'a circle')
        negative_prompt = getMapItem(input_data, 'negative_prompt', 'a circle')
        num_inference_steps = getMapItem(input_data, 'steps', 20)
        num_images_per_prompt = getMapItem(input_data, 'count', 20)
        width = getMapItem(input_data, 'width', 512)
        height = getMapItem(input_data, 'height', 512)
        generator = torch.Generator(device='cuda').manual_seed(input_data["seed"])

        if len(controlnet_images) > 0:
            init_image = controlnet_images

        #
        params = {}
        params['prompt'] = prompt
        params['negative_prompt'] = negative_prompt
        params['num_inference_steps'] = num_inference_steps
        params['num_images_per_prompt'] = num_images_per_prompt
        params['width'] = width
        params['height'] = height
        params['generator'] = generator
        params['height'] = height

        if init_image is not None:
            params['image'] = init_image

        #
        print('====== start inference ======')
        images = model(
            **params
        ).images
        print('====== end inference ======')

        #
        for image in images:
            bucket, key = get_bucket_and_key(output_s3uri)
            key = f'{key}{uuid.uuid4()}.jpg'
            buf = io.BytesIO()

            image.save(buf, format='JPEG')

            s3_client.put_object(
                Body=buf.getvalue(),
                Bucket=bucket,
                Key=key,
                ContentType='image/jpeg',
                Metadata={
                    # #s3 metadata only support ascii
                    "prompt": input_data["prompt"],
                    "seed": str(input_data["seed"])
                }
            )
            print('image: ', f's3://{bucket}/{key}')
            prediction.append(f's3://{bucket}/{key}')

    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        print(f"=================Exception================={ex}")

    print('prediction: ', prediction)
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print(content_type)
    return json.dumps(
        {
            'result': prediction
        }
    )