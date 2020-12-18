import cv2, torch
import numpy as np
from Model import *
from faceRec import *
from face_to_emoji import *
from matplotlib import pyplot

def imgTensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)

def predict(x):
    out = model(imgTensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    label_num = torch.argmax(scaled).item()
    return {'label': label, 'probability': prob, 'index': label_num}
    
    
def batch(size):
    npzfile = np.load("./read_images/raf_db.npz")
    npzfile1 = np.load("./read_images/toronto_face.npz")
    
    train_images = npzfile["inputs_train"]
    train_labels = np.argmax(npzfile["target_train"], axis=1)
    test_images = npzfile["inputs_valid"]
    test_labels = np.argmax(npzfile["target_valid"], axis=1)
    
    train_images1 = npzfile1["inputs_train"]
    train_labels1 = npzfile1["target_train"]
    test_images1 = npzfile1["inputs_valid"]
    test_labels1 = npzfile1["target_valid"]
    
    
    for i in range(len(train_labels1)):
        if train_labels1[i] == 0:
            train_labels1[i] = 5
        elif train_labels1[i] == 1:
            train_labels1[i] = 2
        elif train_labels1[i] == 2:
            train_labels1[i] = 1
        elif train_labels1[i] == 3:
            train_labels1[i] = 3
        elif train_labels1[i] == 4:
            train_labels1[i] = 4
        elif train_labels1[i] == 5:
            train_labels1[i] = 0
        elif train_labels1[i] == 6:
            train_labels1[i] = 6
        else:
            print("wrong train label",train_labels1[i])
        
    for i in range(len(test_labels1)):
        if test_labels1[i] == 0:
            test_labels1[i] = 5
        elif test_labels1[i] == 1:
            test_labels1[i] = 2
        elif test_labels1[i] == 2:
            test_labels1[i] = 1
        elif test_labels1[i] == 3:
            test_labels1[i] = 3
        elif test_labels1[i] == 4:
            test_labels1[i] = 4
        elif test_labels1[i] == 5:
            test_labels1[i] = 0
        elif test_labels1[i] == 6:
            test_labels1[i] = 6
        else:
            print("wrong test label",test_labels1[i])
    
    train_images = np.concatenate((train_images, train_images1))
    train_labels = np.concatenate((train_labels, train_labels1))
    test_images = np.concatenate((test_images, test_images1))
    test_labels = np.concatenate((test_labels, test_labels1))
    
    random_permutation = np.random.permutation(train_images.shape[1])
    for i in range(size):
        pyplot.subplot(4, 30, i+1)
        pyplot.imshow(train_images[random_permutation[i]].reshape((48,48)),interpolation='nearest')
        pyplot.axis('off')
    pyplot.savefig("test.png", bbox_inches='tight')
    pyplot.show()
    
    
if __name__ == "__main__":
    #batch(120)
    model = Model(1, 7)
    softmax = torch.nn.Softmax(dim=1)
    model.load_state_dict(torch.load('9.pth', map_location=get_default_device()))
    out, faces = FaceRec("test2.JPG")
    image = cv2.imread("test2.JPG")
    for i, face in zip(out, faces):
        img = torch.from_numpy(i.reshape((48, 48)))
        img = imgTensor(img)
        prediction = predict(img)
        index = prediction['index']
        image = face_to_emoji(image, face, index)
        print(prediction['label'], prediction['probability'])
    cv2.imshow("face_to_emoji", image)
    cv2.imwrite("face_to_emoji.jpg", image)
    cv2.waitKey(0)
