from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mtcnn import MTCNN
import cv2
import numpy as np
import math

def plot_bounding_boxes(img, result_list):
    plt.imshow(img)
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for _, value in result['keypoints'].items():
            ax.add_patch(Circle(value, radius=1, color='red'))
    plt.show()

def plot_faces(img, result_list):
    for i, result in enumerate(result_list):
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(result_list), i+1)
        plt.axis('off')
        plt.imshow(img[y1:y2, x1:x2])
    plt.show()

def extract_and_save_faces(img, result_list):
    output = []
    for result in result_list:
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        cropped_img = img[y1:y2, x1:x2]
        resized_img = cv2.resize(cropped_img, (48, 48), interpolation=cv2.INTER_CUBIC)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        output.append(gray_img.flatten())
    return np.array(output)

def calculate_angles(faces):
    angles = []
    for face in faces:
        reye = face['keypoints']['right_eye']
        leye = face['keypoints']['left_eye']
        delta_x = reye[0] - leye[0]
        delta_y = reye[1] - leye[1]
        angle = math.atan2(delta_y, delta_x) * 180.0 / math.pi
        angles.append(angle)
    return angles


def face_recognition(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(img_rgb)
    plot_bounding_boxes(img_rgb, faces)
    plot_faces(img_rgb, faces)
    output = extract_and_save_faces(img_rgb, faces)
    angles = calculate_angles(faces)
    return output, faces, angles

def face_to_emoji(img, face, index, angle):
    path = './emojis/' + str(index) + '.png'
    x, y, w, h = face['box']
    y = max(0, y)
    x = max(0, x)
    roi = img[y:y+h, x:x+h]
    h, w = roi.shape[0], roi.shape[1]
    l = min(h, w)
    roi = img[y:y+l, x:x+l]

    # # Adjust for the correct rotation direction
    # Read and rotate emoji
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    emoji_resized = cv2.resize(emoji, (l, l), interpolation=cv2.INTER_AREA)
    matrix = cv2.getRotationMatrix2D((l / 2, l / 2), -angle, 1)
    emoji_rotated = cv2.warpAffine(emoji_resized, matrix, (l, l))

    # Combine emoji with ROI
    # Assuming emoji is a 4-channel image (including alpha)
    for i in range(l):
        for j in range(l):
            if emoji_rotated[i, j][3] != 0:
                roi[i, j] = emoji_rotated[i, j][:3]

    img[y:y+l, x:x+l] = roi
    return img


if __name__ == '__main__':
    output, faces, angles = face_recognition("./test2.jpg")

