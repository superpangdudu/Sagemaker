import cv2
import numpy as np

# 对彩色图像进行Floyd-Steinberg抖动
def floyd_steinberg_dithering(image, levels):
    h, w, c = image.shape
    output = np.zeros((h, w, c), dtype=np.uint8)
    error = np.zeros((h+1, w+1, c), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            old_value = image[y, x].astype(float)
            new_value = np.round(levels * old_value / 256) * (256 / levels)
            output[y, x] = new_value.astype(np.uint8)
            error[y, x] = old_value - new_value
            if x < w-1:
                error[y, x+1] += 7/16 * error[y, x]
                error[y, x+1] = np.clip(error[y, x+1], -new_value, 255-new_value)
            if x > 0 and y < h-1:
                error[y+1, x-1] += 3/16 * error[y, x]
                error[y+1, x-1] = np.clip(error[y+1, x-1], -new_value, 255-new_value)
            if y < h-1:
                error[y+1, x] += 5/16 * error[y, x]
                error[y+1, x] = np.clip(error[y+1, x], -new_value, 255-new_value)
            if x < w-1 and y < h-1:
                error[y+1, x+1] += 1/16 * error[y, x]
                error[y+1, x+1] = np.clip(error[y+1, x+1], -new_value, 255-new_value)
    image += error[:-1, :-1]
    return output

# 加载原始彩色图像
image = cv2.imread('e:/a.jpg')

# 对彩色图像进行Floyd-Steinberg抖动
output = floyd_steinberg_dithering(image.astype(float), 4)

# 显示结果
cv2.imshow('Original', image)
cv2.imshow('Floyd-Steinberg Dithering', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
