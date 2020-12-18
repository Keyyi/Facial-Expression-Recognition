from faceRec import *

import cv2
import numpy as np
from matplotlib import pyplot as plt


def face_to_emoji(path):
    img = cv2.imread(path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    
    for face in faces:
        x, y, w, h = face['box']
        emoji1 = cv2.imread('./emojis/3.png', cv2.IMREAD_UNCHANGED)
        emoji1 = cv2.resize(emoji1, (h, h))
        emoji = emoji1[:, :, :3]
        mask = emoji1[:,:,3]
        mask_inv = 255 - mask
        roi = img[y:y+h, x:x+h]
        img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        emoji_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
        dst = cv2.add(img_bg, emoji_fg)
        img[y:y+h, x:x+h] = dst
    cv2.imshow('res', img)
    cv2.waitKey()
    cv2.destroy()
    
    #bouding_boxes(path, faces)
    #draw_faces(path, faces)
    #output = save_faces(path, faces)
    #return output

if __name__ == '__main__':
    face_to_emoji('./test2.jpg')