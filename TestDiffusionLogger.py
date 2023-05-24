
import diffusers
from diffusers.utils import logging as diffuser_logging
import logging

from diffusers import StableDiffusionPipeline
import torch




#########################################################################################
class LogFilter(logging.Filter):
    def __init(self):
        super(LogFilter, self).__init__()

    def filter(self, record):
        return True


class LogHandler(logging.Handler):
    def __init(self):
        super(LogHandler, self).__init__()

    def emit(self, record):
        value = self.format(record)
        print("###################################################")
        print(value)


#########################################################################################
from tqdm.auto import tqdm as std_tqdm

def external_callback(*args, **kwargs):
    # print(args)
    print(kwargs)


class TqdmExt(std_tqdm):
    def update(self, n=1):
        # print('TqdmExt update')
        displayed = super(TqdmExt, self).update(n)
        if displayed:
            external_callback(**self.format_dict)
        return displayed

#########################################################################################
from diffusers import DiffusionPipeline

def my_progress_bar(self, iterable=None, total=None):
    if not hasattr(self, "_progress_bar_config"):
        self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
        raise ValueError(
            "invalid config"
        )

    if iterable is not None:
        return TqdmExt(iterable, **self._progress_bar_config)
    elif total is not None:
        return TqdmExt(total=total, **self._progress_bar_config)
    else:
        raise ValueError("Either `total` or `iterable` has to be defined.")

DiffusionPipeline.progress_bar = my_progress_bar

#########################################################################################
diffusers.logging.set_verbosity_debug()
logger = diffuser_logging.get_logger("diffusers")

diffusers.logging.disable_default_handler()

handler = LogHandler()
logger.addHandler(handler)

model_id = "/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt"
#model_id = "E:/test/ai-model/stable-diffusion/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "girl playing on the beach, face, smiling, jumping"
negative_prompt = "ugly"

image = pipe(prompt, negative_prompt = negative_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("dudu.png")
