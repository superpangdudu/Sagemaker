import cv2
import numpy as np
import pywt

def downsample_image(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 对每个通道应用小波变换并归一化
    coeffs_r = pywt.dwt2(img_rgb[:, :, 0].astype(np.float32), 'haar')
    coeffs_g = pywt.dwt2(img_rgb[:, :, 1].astype(np.float32), 'haar')
    coeffs_b = pywt.dwt2(img_rgb[:, :, 2].astype(np.float32), 'haar')

    # 只保留低频分量 (LL) 并归一化
    LL_r, (_, _, _) = coeffs_r
    LL_r = cv2.normalize(LL_r, None, 0, 255, cv2.NORM_MINMAX)
    LL_g, (_, _, _) = coeffs_g
    LL_g = cv2.normalize(LL_g, None, 0, 255, cv2.NORM_MINMAX)
    LL_b, (_, _, _) = coeffs_b
    LL_b = cv2.normalize(LL_b, None, 0, 255, cv2.NORM_MINMAX)

    # 将低频分量合并成一个图像
    img_downsampled = np.zeros((LL_r.shape[0], LL_r.shape[1], 3), dtype=np.uint8)
    img_downsampled[:, :, 0] = LL_r
    img_downsampled[:, :, 1] = LL_g
    img_downsampled[:, :, 2] = LL_b

    # 将图像从RGB颜色空间转换回BGR
    img_downsampled = cv2.cvtColor(img_downsampled, cv2.COLOR_RGB2BGR)

    # 保存降采样后的图像
    cv2.imwrite(output_path, img_downsampled)


# 调用这个函数
downsample_image('e:/a.jpg', 'e:/a1.jpg')
downsample_image('e:/a1.jpg', 'e:/a1.jpg')
