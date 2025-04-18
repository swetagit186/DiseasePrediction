import pydicom
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import MyModel  # your model class

def load_dicom_image(file):
    dicom = pydicom.dcmread(file)
    image = dicom.pixel_array.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # normalize
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    model = MyModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, 1).item()
    return predicted
