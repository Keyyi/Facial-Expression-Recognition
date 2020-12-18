import cv2, torch
import numpy as np
from Model import *
from faceRec import *

def img2tensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)


def predict(x):
    out = model(img2tensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    print(label)
    return {'label': label, 'probability': prob}
    
if __name__ == "__main__":
    print("cxnb")
    model = FERModel(1, 7)
    softmax = torch.nn.Softmax(dim=1)
    model.load_state_dict(torch.load('9.pth', map_location=get_default_device()))
    out = FaceRec("test2.jpg")
    for i in out:
        img = torch.from_numpy(i.reshape((48, 48)))
        img = img2tensor(img)
        prediction = predict(img)
        print(prediction['label'], prediction['probability'])


