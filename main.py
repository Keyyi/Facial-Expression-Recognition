import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import *
from faceRec import *
from torchvision import transforms

def img_tensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)

def predict(x):
    out = model(img_tensor(x)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    return {'label': label, 'probability': prob, 'index': torch.argmax(scaled).item()}

def batch(size):
    npzfile = np.load("./read_images/raf_db.npz")
    npzfile1 = np.load("./read_images/toronto_face.npz")

    train_images = np.concatenate((npzfile["inputs_train"], npzfile1["inputs_train"]))
    train_labels = np.concatenate((np.argmax(npzfile["target_train"], axis=1), npzfile1["target_train"]))
    test_images = np.concatenate((npzfile["inputs_valid"], npzfile1["inputs_valid"]))
    test_labels = np.concatenate((np.argmax(npzfile["target_valid"], axis=1), npzfile1["target_valid"]))

    # Label remapping
    label_mapping = {0: 5, 1: 2, 2: 1, 3: 3, 4: 4, 5: 0, 6: 6}
    train_labels = np.vectorize(label_mapping.get)(train_labels)
    test_labels = np.vectorize(label_mapping.get)(test_labels)

    # Shuffle and display images
    random_permutation = np.random.permutation(len(train_images))
    plt.figure(figsize=(15, 5))  # Adjust the figure size if needed
    for i in range(min(size, len(train_images))):
        plt.subplot(4, 30, i + 1)
        plt.imshow(train_images[random_permutation[i]].reshape((48, 48)), interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    batch(120)

    model = Model(1, 7)
    softmax = torch.nn.Softmax(dim=1)
    model.load_state_dict(torch.load('9.pth', map_location=get_default_device()))
    out, faces, angles = face_recognition("test2.jpg")
    image = cv2.imread("test2.jpg")

    for data, face, angle in zip(out, faces, angles):
        img = torch.from_numpy(data.reshape((48, 48)))
        prediction = predict(img)
        image = face_to_emoji(image, face, prediction['index'], angle)
        print(prediction['label'], prediction['probability'])
    
    cv2.imshow("face_to_emoji", image)
    cv2.imwrite("face_to_emoji.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()