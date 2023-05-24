
import numpy as np

configs = [
    ('a', 1, 3),
    ('b', 1, 2),
    ('c', 1, 4)
]

x = []
y = []
def getList(idx):
    name, start, end = configs[idx]
    if idx == 0:
        for i in range(start, end):
            x.append(name + str(i))
            getList(idx + 1)
    else:
        n = x.pop()
        if idx == len(configs) - 1:
            for i in range(start, end):
                y.append(n + '_' + name + str(i))
        else:
            for i in range(start, end):
                x.append(n + '_' + name + str(i))
                getList(idx + 1)

#########################################################################################
controlnet_init_scale = 0.1
controlnet_max_scale = 1.5
controlnet_lineart = None
controlnet_depth = None
controlnets_config = [
    (controlnet_lineart, controlnet_init_scale, controlnet_max_scale, "lineart"),
    (controlnet_depth, controlnet_init_scale, controlnet_max_scale, "depth")
]

controlnets = []
for i in range(len(controlnets_config)):
    c, _, _, _ = controlnets_config[i]
    controlnets.append(c)

controlnet_params = []
tmp = []

def getControlnetConfigs(idx):
    if idx >= len(controlnets_config):
        return

    c, start, end, name = controlnets_config[idx]
    a = np.arange(start, end + 0.1, 0.1)
    a = np.around(a, decimals=1)

    if idx == 0:
        for i in a:
            tmp.append((name + '_' + str(i), [i]))
            getControlnetConfigs(idx + 1)
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

getControlnetConfigs(0)
n = 0


