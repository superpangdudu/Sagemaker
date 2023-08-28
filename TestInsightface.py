
import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image

if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    # swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    swapper = insightface.model_zoo.get_model('z:/download/inswapper_128.onnx')

    # load source face
    source_file = 'e:/a.jpg'
    source_img = cv2.imread(source_file)
    source_faces = app.get(source_img)
    source_face = source_faces[0]

    # b = source_face.bbox
    # i = Image.open(source_file)
    # cropped = i.crop(b)
    # cropped.show()

    #
    dest_file = 'e:/c.jpg'
    dest_img = cv2.imread(dest_file)
    dest_faces = app.get(dest_img)

    res = dest_img.copy()
    for face in dest_faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("e:/swapped.jpg", res)

	# image_file = '/content/.insightface/input/t1.jpg'
    #
	# img = cv2.imread(image_file)
    # faces = app.get(img)
    # faces = sorted(faces, key = lambda x : x.bbox[0])
    # assert len(faces)==6
    # source_face = faces[2]
    # res = img.copy()
    # for face in faces:
    #     res = swapper.get(res, face, source_face, paste_back=True)
    # cv2.imwrite("./t1_swapped.jpg", res)
    # res = []
    # for face in faces:
    #     _img, _ = swapper.get(img, face, source_face, paste_back=False)
    #     res.append(_img)
    # res = np.concatenate(res, axis=1)
    # cv2.imwrite("./t1_swapped2.jpg", res)