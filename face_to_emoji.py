from faceRec import *
import cv2
import numpy as np


def face_to_emoji(img, face, index):
    
    path = './emojis/' + str(index) + '.png'
    x, y, w, h = face['box']
    # First initialize the region of interests
    # with h. Then make sure that roi does not 
    # exceed the boundary of the image.
    roi = img[y:y+h, x:x+h]
    h, w= roi.shape[0], roi.shape[1]
    l = min(h, w)
    roi = img[y:y+l, x:x+l]
    emoji1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    emoji1 = cv2.resize(emoji1, (l, l))
    emoji = emoji1[:, :, :3]
    mask = emoji1[:,:,3]
    mask_inv = 255 - mask
    img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    emoji_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
    dst = cv2.add(img_bg, emoji_fg)
    img[y:y+l, x:x+l] = dst   
    
    return img
    
