
model_config = {
    'base_model' : {
        'name' : 'sd_v15',
        'path' : 's3://models/base/sd_v15.tar.gz'},
    'controlnets' : [
        {
            'name' : 'depth',
            'path' : 's3://models/controlnet/depth.tar.gz',
            'scale' : 0.8
        },
        {
            'name' : 'lineart',
            'path' : 's3://models/controlnet/lineart.tar.gz',
            'scale' : 0.7
        }
    ],
    'loras' : [
        {
            'name' : 'handPaintedPortrait_v12',
            'path' : 's3://models/lora/handPaintedPortrait_v12.tar.gz',
            'scale' : 0.8
        }
    ]
}

#
BASE_MODEL_NAME = model_config['base_model']['name']
BASE_MODEL_PATH = model_config['base_model']['path']

#
CONTROLNET_MODELS_NAME = []
CONTROLNET_MODELS_PATH = []
CONTROLNET_MODELS_SCALE = []

controlnets = model_config['controlnets']
for i in range(len(controlnets)):
    CONTROLNET_MODELS_NAME.append(controlnets[i]['name'])
    CONTROLNET_MODELS_PATH.append(controlnets[i]['path'])
    CONTROLNET_MODELS_SCALE.append(controlnets[i]['scale'])

#
LORA_MODELS_NAME = []
LORA_MODELS_PATH = []
LORA_MODELS_SCALE = []

loras = model_config['loras']
for i in range(len(loras)):
    LORA_MODELS_NAME.append(loras[i]['name'])
    LORA_MODELS_PATH.append(loras[i]['path'])
    LORA_MODELS_SCALE.append(loras[i]['scale'])

n = 0