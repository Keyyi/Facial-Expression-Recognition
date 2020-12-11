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

model = FERModel(1, 7)
softmax = torch.nn.Softmax(dim=1)
model.load_state_dict(torch.load('Resnet9.pth', map_location=get_default_device()))

def predict(x):
    out = model(img2tensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    return {'label': label, 'probability': prob}
    
if __name__ == "__main__":
    img = torch.from_numpy(FaceRec("wbb.jpg"))
    img = img2tensor(img)
    prediction = predict(img)
    
    print(prediction['label'], prediction['probability'])


