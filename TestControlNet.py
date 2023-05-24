
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from controlnet_aux import OpenposeDetector


#########################################################################################
def image2Canny(inputImg, outputImg):
    image = Image.open(inputImg)
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    canny_image = Image.fromarray(image)
    canny_image.save(outputImg)


def cannyImage2Attention(cannyImage, outputImg):
    cannyImage = Image.open(cannyImage)
    cannyImage = np.array(cannyImage)
    canny_image = Image.fromarray(cannyImage)

    generator = torch.Generator(device='cuda').manual_seed(12345)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    #
    lora_path = '/home/zlkh000/krly/lifei/ai-models/sd-dudu-head-model-lora'
    pipe.unet.load_attn_procs(lora_path)
    pipe.to("cuda")

    #
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    prompt = 'a dog'
    image = pipe(prompt,
                 num_inference_steps=20,
                 generator=generator,
                 image=canny_image,
                 controlnet_conditioning_scale=0.5).images[0]
    image.save(outputImg)

    image = pipe('a cat',
                 num_inference_steps=20,
                 generator=generator,
                 image=canny_image,
                 controlnet_conditioning_scale=0.5).images[0]
    image.save('x.jpg')



def getImagePose(imgIn, imgOut):
    image = Image.open(imgIn)
    pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    poseImg = pose(image)
    poseImg.save(imgOut)


def imageFromPose(poseImg, imgOut):
    pose = Image.open(poseImg)
    pose = np.array(pose)
    pose = Image.fromarray(pose)

    generator = torch.Generator(device='cuda').manual_seed(12345)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    prompt = "a beautiful hollywood actress wearing black dress attending award winning event, red carpet stairs at background"
    prompt = 'a batman'
    negativePrompt = ''
    image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=pose,
        controlnet_conditioning_scale=1.0
    ).images[0]
    image.save(imgOut)


def imageFromScribble(scribbleImg, imgOut):
    scribble = Image.open(scribbleImg)
    scribble = np.array(scribble)
    scribble = Image.fromarray(scribble)

    generator = torch.Generator(device='cuda').manual_seed(12345)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    prompt = "a beautiful hollywood actress wearing black dress attending award winning event, red carpet stairs at background"
    prompt = 'a batman'
    negativePrompt = ''
    image = pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=scribble,
        controlnet_conditioning_scale=1.0
    ).images[0]
    image.save(imgOut)


inputImage = 'e:/a.jpg'
outputImage = 'e:/canny-of-original.jpg'
#########################################################################################
if __name__ == '__main__':
    #image2Canny(inputImage, outputImage)
    #cannyImage2Attention(outputImage, 'attention.jpg')
    #getImagePose(inputImage, 'pose.jpg')
    #imageFromPose('pose.jpg', 'poseOut.jpg')
    imageFromScribble('scribble.jpg', 'scribbleOut.jpg')

