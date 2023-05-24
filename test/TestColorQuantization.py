import cv2
import numpy as np

# 对彩色图像进行色彩量化和抖动
def color_quantization_and_dithering(image, levels):
    h, w, c = image.shape
    output = np.zeros((h, w, c), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            old_value = image[y, x].astype(float)
            new_value = np.round(levels * old_value / 256) * (256 / levels)
            output[y, x] = new_value.astype(np.uint8)
            error = old_value - new_value
            if x < w-1:
                image[y, x+1] += 7/16 * error
            if x > 0 and y < h-1:
                image[y+1, x-1] += 3/16 * error
            if y < h-1:
                image[y+1, x] += 5/16 * error
            if x < w-1 and y < h-1:
                image[y+1, x+1] += 1/16 * error
    return output

# 加载原始彩色图像
image = cv2.imread('e:/a.jpg')

# 对彩色图像进行色彩量化和抖动
output = color_quantization_and_dithering(image.astype(float), 4)

# 显示结果
cv2.imshow('Original', image)
cv2.imshow('Color Quantization and Dithering', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
