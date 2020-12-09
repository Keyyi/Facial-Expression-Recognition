from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

def draw_image_with_boxes(filename, result_list):
    img = pyplot.imread(filename)
    pyplot.imshow(img)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    pyplot.show()
 
def draw_faces(filename, result_list):
    data = pyplot.imread(filename)
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        pyplot.imshow(data[y1:y2, x1:x2])
    pyplot.show()
    
def FaceRec(path):
    img = pyplot.imread(path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    draw_image_with_boxes(path, faces)
    draw_faces(path, faces)

if __name__ == '__main__':
    FaceRec("./test2.jpg")

