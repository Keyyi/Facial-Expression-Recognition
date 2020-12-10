from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

def draw_image_with_boxes(path, result_list):
    img = pyplot.imread(path)
    pyplot.imshow(img)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            ax.add_patch(Circle(value, radius=1, color='red'))
    pyplot.show()
 
def draw_faces(path, result_list):
    data = pyplot.imread(path)
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        pyplot.imshow(data[y1:y2, x1:x2])
    pyplot.show()

def save_faces(path, result_list):
    data = pyplot.imread(path)
    output = np.empty((len(result_list), 48*48))
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        img = data[y1:y2, x1:x2]
        res = cv2.resize(img, dsize = (48, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        output[i] = gray.reshape((1, 48*48))
        pyplot.imshow(res)
    pyplot.show()    
    return output
    
def FaceRec(path):
    img = pyplot.imread(path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    draw_image_with_boxes(path, faces)
    draw_faces(path, faces)
    output = save_faces(path, faces)
    return output

if __name__ == '__main__':
    output = FaceRec("./test3.jpg")
    

