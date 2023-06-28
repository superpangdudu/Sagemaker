

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
            'scale_to': 1.0
        },
        {
            'name': 'lineart',
            'path': 'lllyasviel/control_v11p_sd15_lineart',
            'scale_from': 0.8,
            'scale_to': 1.8
        },
        {
            'name': 'x',
            'path': 'lllyasviel/control_v11p_sd15_lineart',
            'scale_from': 0.8,
            'scale_to': 1.2
        }
    ],
    'loras': [
        {
            'name': 'add_detail',
            'path': '',
            'scale_from': 0.8,
            'scale_to': 0.8
        },
        {
            'name': 'epi_noiseoffset2',
            'path': '',
            'scale_from': 0.8,
            'scale_to': 0.8
        }
    ]
}


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
        pass

    def get_controlnet_image(self, img):
        return img

    def get_scale(self):
        return self.scale_from, self.scale_to


class LineartControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        super().__init__(model_path, 'lineart', s_from, s_to)

    def get_controlnet_image(self, img):
        return img


class DepthControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        super().__init__(model_path, 'depth', s_from, s_to)

    def get_controlnet_image(self, img):
        return img


class TileControlnetConfig(ControlnetConfig):
    def __init__(self, model_path, s_from, s_to):
        super().__init__(model_path, 'tile', s_from, s_to)

    def get_controlnet_image(self, img):
        return img


def get_controlnet(cfg_name, model_path, s_from, s_to):
    if cfg_name == 'lineart':
        return LineartControlnetConfig(model_path, s_from, s_to)
    elif cfg_name == 'depth':
        return DepthControlnetConfig(model_path, s_from, s_to)
    elif cfg_name == 'tile':
        return TileControlnetConfig(model_path, s_from, s_to)

    return None


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


make_inference_scale_params(controlnet_scale_list)


#
lora_scale_list = []
for i in range(len(model_config['loras'])):
    name = model_config['loras'][i]['name']
    path = model_config['loras'][i]['path']
    scale_from = model_config['loras'][i]['scale_from']
    scale_to = model_config['loras'][i]['scale_to']

    lora_scale_list.append((name, scale_from, scale_to))



