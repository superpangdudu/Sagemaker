
from diffusers import StableDiffusionPipeline
import torch

pretrained_v15_path = '/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt'
dataset = '/home/zlkh000/krly/lifei/test-images/dudu-head-processed'


def txt2img(sd_pipe, p, np, steps=50, gs=7.5, count=1, prefix='test'):
    for i in range(count):
        image = sd_pipe(prompt=p,
                        negative_prompt=np,
                        num_inference_steps=steps,
                        guidance_scale=gs).images[0]
        image.save('out/' + prefix + '-' + str(i) + ".png")


def doSomething():
    model_path = "/home/zlkh000/krly/lifei/ai-models/sd-dudu-head-model-lora"
    pipe = StableDiffusionPipeline.from_pretrained("/home/zlkh000/krly/lifei/ai-models/v15_from_ckpt",
                                                   torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(model_path)
    pipe.to("cuda")

    steps = 100
    gs = 11

    prompt = '''dudu girl playing basketball with micheal jordan in his studio, 4 k photograph. volumetric lighting and shadows!!! by diane arbus!! tiffany bokeh! artgerm lau wlop rossdraws pastel vibrant colors octane render pixar shaded makoto aki & hiromasa ogura toriyama greg rutkowski definition highly detailed intricate matte'''
    negative_prompt = 'ugly, distort, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, poorly drawn face, poorly drawn eyes, poorly drawn nose, poorly drawn mouth, poorly Rendered face, disfigured, deformed body features, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

    style = 'jordan'
    # for i in range(200):
    #         image = pipe(prompt,
    #              negative_prompt=negative_prompt,
    #              num_inference_steps=steps, guidance_scale=gs).images[0]
    #         image.save(style + '-' + str(i) + ".png")

    count = 200

    prompt = '''dudu girl playing basketball with micheal jordan, digital painting masterpiece by frank frazetta and kim jung gi artstyle of a vibrant neo roman color scheme. the background is an amazing sci - fi scenery view from behind on top in front at night sky filled dreamscape like blade runner 2 0 4 9 city canals mountains tokyo photorealism detailed image trending award winning photography cinematic lighting dramatic shadows hdr 8'''
    style = 'jordan1'
    txt2img(pipe, prompt, negative_prompt, steps, gs, count, style)

    #
    prompt = '''dudu girl playing basketball with micheal jordan, digital painting by greg rutkowski and alphonse mucha. highly detailed 8 k artstation hq sharp focus illustration intricate masterpiece fine detail delicate features muted colors red light yellow black background chiaroscuro perfect face award winning victo ngai ilyArajima style beeple beautiful stylized bold thomas eakins in the dark forest concept artwork for'''
    style = 'jordan2'
    txt2img(pipe, prompt, negative_prompt, steps, gs, count, style)

    #
    prompt = '''dudu girl playing basketball with micheal jordan, oil on canvas by edward hopper and rene magritte. 8k resolution trending at artstation conceptart digital illustration cel shading behance hdr painting of a cyberpunk cityscape in the distance overgrown jungle landscape isometric metropolis filmic photoshop realism detailed matte background vibrant nature hyperrealism retrofuturist vaporwave colors scheme pastel neon color palette'''
    style = 'jordan3'
    txt2img(pipe, prompt, negative_prompt, steps, gs, count, style)

    #
    prompt = '''dudu girl playing basketball with micheal jordan, digital illustration by artgerm and karol bak. highly detailed 8k resolution hdr smooth sharp focus high contrast lighting trending on deviantart hyperrealism fanzine unreal engine 5 professional photography portrait of beautiful face in the style dslr camera black background bokeh dof shrek dramatic lightning cinematic movie shot from a distance realistic volumetric studio lights 3'''
    style = 'jordan4'
    txt2img(pipe, prompt, negative_prompt, steps, gs, count, style)

    #
    prompt = '''a (dudu girl) as king of pop performing on a big stage, highly detailed painting by gaston bussiere craig mullins j. c leyendecker gustav klimt artgerm greg rutkowski john berkey 8k octane render redshift unreal engine key lights trending on pixiv hyperrealism oil canvas depth of field bokeh effect very coherent symmetrical artwork high detail full body character drawing sharp focus closeup'''
    style = 'pop6'
    txt2img(pipe, prompt, negative_prompt, steps, gs, count, style)

    #
    # prompt = '(dudu girl) by Makoto Shinkai, face to camera, trending on artstation. unreal engine 5 rendered with redshift render and path tracing extremely detailed 8k post processing very sharp quality 3D model cg blender cycles maxwell bokeh pixar mayan Engine smooth portrait hd 4K full body character concept octane renderer in the style of Pixar Ghibli studio key shot loish beeple dan mumford global illumination ray traced lighting vector'
    # #prompt = 'dudu girl by Alphonse Mucha'
    # style = 'MakotoShinkai'
    # for i in range(200):
    #     image = pipe(prompt,
    #              negative_prompt=negative_prompt,
    #              num_inference_steps=steps, guidance_scale=gs).images[0]
    #     image.save('out/' + style + '-' + str(i) + ".png")
    #
    # prompt = 'a dudu girl riding panda on moon, face to camera, 4k wallpaper. highly detailed digital art trending on ArtStation and unreal engine 5 rendered with octane render volumetric cinematic light cgi rtx hdr style chrome reflexion glowing fog raytracing glow fanart arstation pixiv contest winner 3D behance HD 8K ultra high resolution photo realistic illustration dramatic lighting sharpness global illumination subsurface scattering shadows bokeh HDR stylized dark'
    # image = pipe(prompt,
    #              negative_prompt=negative_prompt,
    #              num_inference_steps=steps, guidance_scale=gs).images[0]
    # image.save('out/' + "3.png")
    #
    # prompt = 'a dudu girl by Makoto Shinkai, face to camera, concept art of the main character in yellow raincoat with hoodie and sunglasses holding a gun dramatic lighting trending on ArtStation HQ 8K octane render ultra high intricate details digital anime cgstation illustration sharp detailed face shot close-up portrait cinematic lightning from above view waist up smiling official media made out painting oilpaint canvas paint splashes many colors unreal engine 4k post processing very realistic detail photorealistic'
    # image = pipe(prompt,
    #              negative_prompt=negative_prompt,
    #              num_inference_steps=steps, guidance_scale=gs).images[0]
    # image.save('out/' + "4.png")

if __name__ == '__main__':
    doSomething()
