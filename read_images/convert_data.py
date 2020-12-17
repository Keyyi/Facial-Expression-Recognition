import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2

inputs_train = np.zeros((12271, 2304))
target_train = np.zeros((12271, 7))
inputs_test = np.zeros((3068, 2304))
target_test = np.zeros((3068, 7))

file = open('list_patition_label.txt', 'r')
lines = file.readlines() 
i = 0
j = 0

def resize_faces(path, result_list):
    data = pyplot.imread(path)
    h, w = data.shape[0], data.shape[1]
    output = np.empty((1, 48*48))
    flag = 0
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        x1 = max(0, x1)
        x2 = min(x2, w)
        y1 = max(0, y1)
        y2 = min(y2, h)
        if (width < w/2 or height < h/2):
            flag = 1
            break
        img = data[y1:y2, x1:x2]
        res = cv2.resize(img, dsize = (48, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        output = gray.reshape((1, 48*48))
        break
    
    # For some cases(e.g. wearing glasses) the MTCNN can not detect faces,
    # or it detect something that's really small, we simply resize the image
    if (len(result_list) == 0 or flag ==1):
        res = cv2.resize(data, dsize = (48, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        output = gray.reshape((1, 48*48))       
        
    return output

    
def FaceRec(path):
    img = pyplot.imread(path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    output = resize_faces(path, faces)
    return output

for line in lines:
    name, label = line.strip('\n').split(' ')
    pre, suf = name.split('.')
    path = 'aligned/' + pre + '_aligned.' + suf
    index = int(label) -1
    if (i < 12271):
        inputs_train[i] = FaceRec(path)
        target_train[i][index] = 1
        i += 1
    else:
        inputs_test[j] = FaceRec(path)
        target_test[j][index] = 1
        j += 1        

np.savez('raf_train_db',inputs_train=inputs_train,target_train=target_train) 
np.savez('raf_test_db', inputs_test=inputs_test, target_test = target_test)