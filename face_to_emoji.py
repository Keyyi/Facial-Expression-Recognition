from faceRec import *
import cv2
import numpy as np


def face_to_emoji(img, face, index):
    
    path = './emojis/' + str(index) + '.png'
    x, y, w, h = face['box']
    emoji1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    emoji1 = cv2.resize(emoji1, (h, h))
    emoji = emoji1[:, :, :3]
    mask = emoji1[:,:,3]
    mask_inv = 255 - mask
    roi = img[y:y+h, x:x+h]    
    img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    emoji_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
    dst = cv2.add(img_bg, emoji_fg)
    img[y:y+h, x:x+h] = dst   
    
    return img
    
