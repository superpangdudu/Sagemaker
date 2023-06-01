import os
import json
import io
import sys
import tarfile
import traceback
import uuid
import subprocess
import shutil

import logging

from PIL import Image

import boto3
import sagemaker
import torch
import s3fs

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import AltDiffusionPipeline
from diffusers import AltDiffusionImg2ImgPipeline

from diffusers import EulerDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import HeunDiscreteScheduler
from diffusers import LMSDiscreteScheduler
from diffusers import KDPM2DiscreteScheduler
from diffusers import KDPM2AncestralDiscreteScheduler
from diffusers import DDIMScheduler

from safetensors.torch import load_file, save_file
from transformers import BlipProcessor, BlipForConditionalGeneration


logging.basicConfig(level=logging.INFO)
#########################################################################################
account_id = boto3.client('sts').get_caller_identity().get('Account')
region_name = boto3.session.Session().region_name

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
role = sagemaker.get_execution_role()

logging.info(f"role: {role}")
logging.info(f"bucket: {bucket}")

#########################################################################################
s3_client = boto3.client('s3')

max_height = os.environ.get("max_height", 768)
max_width = os.environ.get("max_width", 768)
max_steps = os.environ.get("max_steps", 100)
max_count = os.environ.get("max_count", 4)
s3_bucket = os.environ.get("s3_bucket", "")
custom_region = os.environ.get("custom_region", None)
safety_checker_enable = json.loads(os.environ.get("safety_checker_enable", "false"))

schedulers = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "eular": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "ddim": DDIMScheduler
}

# default scheduler
DEFAULT_SCHEDULER = 'euler_a'
# default guidance scale
DEFAULT_GUIDANCE_SCALE = 12
#########################################################################################
'''
Model path
Base model: 'model/base/{model name}'
Controlnet: 'model/controlnet/{controlnet model name}'
Lora: 'model/lora/{LoRA model name}'
'''

model_config = {
    'base_model': {
        'name': 'watercolor_v3',
        'path': f's3://{bucket}/model/base/watercolor_v3',
        'default': 'runwayml/stable-diffusion-v1-5'
    },
    'controlnets': [
        {
            'name': 'tile',
            'path': f's3://{bucket}/model/controlnet/control_v11f1e_sd15_tile',
            'default': 'lllyasviel/control_v11f1e_sd15_tile',
            'scale': 0.7
        },
        {
            'name': 'lineart',
            'path': f's3://{bucket}/model/controlnet/control_v11p_sd15_lineart',
            'default': 'lllyasviel/control_v11p_sd15_lineart',
            'scale': 0.7
        }
    ],
    'loras': [
        # {
        #     'name': 'handPaintedPortrait_v12',
        #     'path': f's3://{bucket}/model/lora/handPaintedPortrait_v12.safetensors',
        #     'scale': 0.8
        # }
    ]
}


#########################################################################################
class ControlnetConfig:
    """
    ControlnetConfig
    """
    def __init__(self, remote_model_path, local_model_path, model_name, controlnet_scale):
        self.remote_model_path = remote_model_path
        self.parent_dir = local_model_path
        self.name = model_name
        self.model_path = local_model_path + model_name
        self.scale = controlnet_scale

    def newControlNetModel(self):
        return ControlNetModel.from_pretrained(self.model_path, torch_dtype=torch.float16)

    def getControlnetImage(self, img):
        return img

    def getScale(self):
        return self.scale

    # TODO prepare model files


class LineartControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, local_model_path, controlnet_scale):
        from controlnet_aux import LineartDetector

        super().__init__(remote_model_path, local_model_path, 'lineart', controlnet_scale)
        self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        logging.info('######################  LineartControlnetConfig.processor done')

    def getControlnetImage(self, img):
        logging.info('######################  LineartControlnetConfig.getControlnetImage start')
        out = self.processor(img)
        logging.info('######################  LineartControlnetConfig.getControlnetImage end')
        return out


class DepthControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, local_model_path, controlnet_scale):
        from transformers import pipeline as transformers_pipeline

        super().__init__(remote_model_path, local_model_path, 'depth', controlnet_scale)
        self.depth_estimator = transformers_pipeline('depth-estimation')
        logging.info('######################  DepthControlnetConfig.depth_estimator done')

    def getControlnetImage(self, img):
        logging.info('######################  DepthControlnetConfig.getControlnetImage start')
        out = self.depth_estimator(img)['depth']
        logging.info('######################  LineartControlnetConfig.getControlnetImage end')
        return out


class TileControlnetConfig(ControlnetConfig):
    def __init__(self, remote_model_path, local_model_path, controlnet_scale):
        super().__init__(remote_model_path, local_model_path, 'tile', controlnet_scale)


def getControlnet(remote_model_path, model_parent_dir, controlnet_name, controlnet_scale):
    if controlnet_name == 'lineart':
        return LineartControlnetConfig(remote_model_path, model_parent_dir, controlnet_scale)
    elif controlnet_name == 'depth':
        return DepthControlnetConfig(remote_model_path, model_parent_dir, controlnet_scale)
    elif controlnet_name == 'tile':
        return TileControlnetConfig(remote_model_path, model_parent_dir, controlnet_scale)

    return None


#########################################################################################
MODEL_ROOT_PATH = '/home/ec2-user/SageMaker/model/'

#
BASE_MODEL_NAME = model_config['base_model']['name']
BASE_MODEL_REMOTE_PATH = model_config['base_model']['path']
BASE_MODEL_DEFAULT_PATH = model_config['base_model']['default']
BASE_MODEL_PATH = MODEL_ROOT_PATH + 'base/'

logging.info('#' * 90)
logging.info('Base model name: %s', BASE_MODEL_NAME)
logging.info('Base model remote path: %s', BASE_MODEL_REMOTE_PATH)
logging.info('Base model local path: %s', BASE_MODEL_PATH)

# prepare controlnet configs
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


# prepare LoRAs
# FIXME
class LoRAConfig:
    lora_name = ''
    remote_model_path = ''
    lora_scale = 0.5


lora_configs = []
loras = model_config['loras']
for i in range(len(loras)):
    cfg = LoRAConfig()
    cfg.lora_name = loras[i]['name']
    cfg.remote_model_path = loras[i]['path']
    cfg.lora_scale = loras[i]['scale']

    lora_configs.append(cfg)

    logging.info('#' * 90)
    logging.info(f'LoRA: {cfg.lora_name}, scale: {cfg.lora_scale}, remote path: {cfg.remote_model_path}')

LORA_ROOT_PATH = MODEL_ROOT_PATH + 'lora/'


#########################################################################################
def download_from_s3(remote, local):
    logging.info(f'########## start download_from_s3, from {remote} to {local}')
    fs = s3fs.S3FileSystem()
    fs.get(remote, local, recursive=True)


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
    # FIXME
    if os.path.isdir(MODEL_ROOT_PATH):
        shutil.rmtree(MODEL_ROOT_PATH)

    os.makedirs(BASE_MODEL_PATH)
    os.makedirs(LORA_ROOT_PATH)

    tmp_zip_path = BASE_MODEL_PATH + 'tmp'  # base/tmp
    os.makedirs(tmp_zip_path)
    os.makedirs(BASE_MODEL_PATH + 'model')

    # base model
    logging.info(f'###### start to download base model from {BASE_MODEL_REMOTE_PATH} to {tmp_zip_path}')
    download_from_s3(BASE_MODEL_REMOTE_PATH, tmp_zip_path)

    # FIXME why the files not downloaded in tmp, but in tmp/{base_model_name}?
    command = f'ls {tmp_zip_path}/{BASE_MODEL_NAME}'
    subprocess.run(command, shell=True)
    print(command)

    logging.info(f'###### start to unzip base model in {tmp_zip_path}')
    command = f"cat {tmp_zip_path}/{BASE_MODEL_NAME}/*.zip > {BASE_MODEL_PATH}model/model.zip"
    subprocess.run(command, shell=True)

    shutil.rmtree(tmp_zip_path)

    command = f"unzip {BASE_MODEL_PATH}model/model.zip -d {BASE_MODEL_PATH}"
    subprocess.run(command, shell=True)

    shutil.rmtree(BASE_MODEL_PATH + 'model')

    logging.info(f'###### base model file is ready, {BASE_MODEL_PATH}{BASE_MODEL_NAME}')

    # controlnets
    for i in range(len(controlnet_configs)):
        controlnet_model_path = CONTROLNET_ROOT_PATH + controlnet_configs[i].name

        logging.info(f'###### start to download controlnet {controlnet_model_path}')
        download_from_s3(controlnet_configs[i].remote_model_path, controlnet_model_path)
        logging.info(f'###### ControlNet {controlnet_model_path} is downloaded')

        command = f'ls {controlnet_model_path}'
        subprocess.run(command, shell=True)
        print(command)

    logging.info('###### controlnet model file is ready')

    # loras
    for i in range(len(lora_configs)):
        lora_model_name = os.path.basename(lora_configs[i].remote_model_path)
        lora_model_path = LORA_ROOT_PATH + lora_model_name

        logging.info(f'###### Start to download LoRA {lora_model_path}')
        download_from_s3(lora_configs[i].remote_model_path, lora_model_path)
        logging.info(f'###### LoRA {lora_model_path} is downloaded')
    logging.info('###### LoRA model file is ready')


#########################################################################################
prepare_models()


#########################################################################################
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
# FIXME
WATER_MARK_IMAGE_REMOTE_PATH = f's3://{bucket}/stablediffusion/watermark/watermark.png'

watermark_image = None
watermark_width, watermark_height = 20, 20


def make_watermark_on_image(img):
    global watermark_image, watermark_width, watermark_height
    if watermark_image is None:
        watermark = os.path.basename(WATER_MARK_IMAGE_REMOTE_PATH)
        local_watermark_path = f'/tmp/{watermark}'
        download_from_s3(WATER_MARK_IMAGE_REMOTE_PATH, local_watermark_path)

        watermark_image = Image.open(local_watermark_path).convert('RGBA')
        watermark_width, watermark_height = watermark_image.size

    image_width, image_height = img.size
    watermark_layer = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    watermark_layer.paste(watermark_image, (image_width - (watermark_width + 10), image_height - (watermark_height + 10)))

    out = Image.composite(watermark_layer, img, watermark_layer)
    return out


#########################################################################################
# controlnet models
controlnets = []
logging.info(f'########## start to create controlnet model')
for i in range(len(controlnet_configs)):
    cn = ControlNetModel.from_pretrained(CONTROLNET_ROOT_PATH + controlnet_configs[i].name,
                                         torch_dtype=torch.float16)
    controlnets.append(cn)

# base model
logging.info(f'########## start to create base model {BASE_MODEL_PATH}{BASE_MODEL_NAME}')

sd_pipeline = None
model_path = BASE_MODEL_PATH + BASE_MODEL_NAME
if len(controlnets) > 0:
    sd_pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_path,
                                                                    controlnet=controlnets,
                                                                    torch_dtype=torch.float16)
else:
    sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

# LoRA models
logging.info(f'########## start to create lora model')
for i in range(len(lora_configs)):
    lora_model_name = os.path.basename(lora_configs[i].remote_model_path)
    model = f"{LORA_ROOT_PATH}{lora_model_name}"

    sd_pipeline = __load_lora(sd_pipeline, model, lora_configs[i].lora_scale)

sd_pipeline.to('cuda')
sd_pipeline.enable_model_cpu_offload()
sd_pipeline.enable_xformers_memory_efficient_attention()
sd_pipeline.enable_attention_slicing()

logging.info('########## models creation done')


#########################################################################################
# BLIP
logging.info('########## start to create BLIP processor')
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

logging.info('########## start to create BLIP model')
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

logging.info('########## BLIP creation done')


#########################################################################################
def model_fn(model_dir):
    logging.info('########## model_fn start ##########')
    logging.info('########## model_fn end ##########')

    return sd_pipeline


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
    logging.info(f"=================input_fn=================\n{request_content_type}\n{request_body}")
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
    bkt = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bkt, key


def prepare_opt(input_data):
    """
    Prepare inference input parameter
    """
    opt = {}
    opt["prompt"] = input_data.get("prompt", "a photo of an astronaut riding a horse on mars")
    opt["negative_prompt"] = input_data.get("negative_prompt", "")
    opt["steps"] = clamp_input(input_data.get("steps", 20), minn=20, maxn=max_steps)
    opt["sampler"] = input_data.get("sampler", "euler_a")
    opt["height"] = clamp_input(input_data.get("height", 512), minn=64, maxn=max_height)
    opt["width"] = clamp_input(input_data.get("width", 512), minn=64, maxn=max_width)
    opt["count"] = clamp_input(input_data.get("count", 1), minn=1, maxn=max_count)
    opt["seed"] = input_data.get("seed", 1024)
    opt["input_image"] = input_data.get("input_image", None)
    opt["guidance_scale"] = input_data.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)

    if 'guess' in input_data:
        opt["guess"] = 1

    if 'watermark' in input_data:
        opt["watermark"] = 1

    if 'blip' in input_data:
        opt["blip"] = 1

    logging.info(f"=================prepare_opt=================\n{opt}")
    return opt


def predict_fn(input_data, m):
    """
    Apply model to the incoming request
    """
    logging.info("=================predict_fn=================")
    logging.info(f'input_data: {input_data}')
    prediction = []

    ##########
    # FIXME a workaround to do some runtime routines
    ##########

    try:
        sagemaker_session = sagemaker.Session() if custom_region is None else sagemaker.Session(boto3.Session(region_name=custom_region))
        bucket = sagemaker_session.default_bucket()
        if s3_bucket != "":
            bucket = s3_bucket

        default_output_s3uri = f's3://{bucket}/stablediffusion/asyncinvoke/images/'

        output_s3uri = input_data.get('output_s3uri', default_output_s3uri)

        infer_args = input_data.get('infer_args', None)
        logging.info(f'infer_args: {infer_args}')

        init_image = None
        input_image = input_data['input_image']
        logging.info(f'init_image: {init_image}')
        logging.info(f'input_image: {input_image}')

        # download initial image from S3
        if input_image is not None:
            image_name = os.path.basename(input_image)
            local_image_path = f'/tmp/{image_name}'
            download_from_s3(input_image, local_image_path)

            init_image = Image.open(local_image_path).convert("RGB")
            init_image = init_image.resize((input_data["width"], input_data["height"]))
            logging.info('init_img download done')

        logging.info(f'init_image2: {init_image}')
        logging.info(f'input_image2: {input_image}')
        # prepare controlnet images
        controlnet_images = []
        for i in range(len(controlnet_configs)):
            img = controlnet_configs[i].getControlnetImage(init_image)
            controlnet_images.append(img)

        # prepare controlnet scales
        controlnet_conditioning_scale = []
        for i in range(len(controlnet_configs)):
            controlnet_scale = controlnet_configs[i].getScale()
            controlnet_conditioning_scale.append(controlnet_scale)

        #
        prompt = input_data.get('prompt', 'a circle')
        negative_prompt = input_data.get('negative_prompt', 'a circle')
        num_inference_steps = input_data.get('steps', 20)
        num_images_per_prompt = input_data.get('count', 2)
        width = input_data.get('width', 512)
        height = input_data.get('height', 512)
        guidance_scale = input_data.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)

        # get prompt by BLIP if necessary
        if 'blip' in input_data:
            blip_input = blip_processor(init_image, return_tensors="pt").to("cuda")
            blip_out = blip_model.generate(**blip_input)
            blip_prompt = blip_processor.decode(blip_out[0], skip_special_tokens=True)

            prompt += f', {blip_prompt}'

            logging.info(f'########## new prompt after BLIP: {prompt}')

        #
        guess_mode = False
        if 'guess' in input_data:
            guess_mode = True

        #
        generator = torch.Generator(device='cuda').manual_seed(input_data["seed"])

        #
        if len(controlnet_images) > 0:
            init_image = controlnet_images

        #
        scheduler = input_data.get('sampler', DEFAULT_SCHEDULER)
        if scheduler is not None:
            scheduler = schedulers.get('scheduler', DDIMScheduler)
            sd_pipeline.scheduler = scheduler.from_config(sd_pipeline.scheduler.config)

        #
        params = {}
        params['prompt'] = prompt
        params['negative_prompt'] = negative_prompt
        params['num_inference_steps'] = num_inference_steps
        params['num_images_per_prompt'] = num_images_per_prompt
        params['width'] = width
        params['height'] = height
        params['generator'] = generator
        params['guidance_scale'] = guidance_scale

        if guess_mode is True:
            params['guess_mode'] = True

        if init_image is not None:
            params['image'] = init_image

        if len(controlnet_conditioning_scale) > 0:
            params['controlnet_conditioning_scale'] = controlnet_conditioning_scale

        logging.info(f'====== params = {params}')

        # do inference
        logging.info('====== start inference ======')
        images = sd_pipeline(
            **params
        ).images
        logging.info('====== end inference ======')

        # save images to S3
        for image in images:
            bucket, key = get_bucket_and_key(output_s3uri)
            key = f'{key}{uuid.uuid4()}.jpg'
            buf = io.BytesIO()

            # for watermark FIXME
            if 'watermark' in input_data:
                image = make_watermark_on_image(image)

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
            logging.info(f'image: s3://{bucket}/{key}')
            prediction.append(f's3://{bucket}/{key}')

    except Exception as ex:
        traceback.print_exc(file=sys.stdout)
        logging.error(f"=================Exception================={ex}")

    logging.info(f'prediction: {prediction}')
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    logging.info(content_type)
    return json.dumps(
        {
            'result': prediction
        }
    )
