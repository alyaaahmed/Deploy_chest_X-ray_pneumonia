import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Conda environment (if active): {os.environ.get('CONDA_PREFIX')}")
print(f"Python version: {sys.version}")

import torch
from torch import nn
from torchvision import transforms , models 
from PIL import Image 
import pickle
device = torch.device("cuda"if torch.cuda.is_available()else "cpu")
model = models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
model.fc= nn.Linear(in_features=512,out_features=2,bias=True) 
model.load_state_dict(torch.load(r"models\model.pth"))
model.to(device)
with open(r"data\label_encoder.pkl" , "rb") as f:
    label_encoder=pickle.load(f)
# Function to make predictions on new images
def infer(img):
    # Convert image to grayscale
    img_gray = img.convert('L')
    
    # Transform image to tensor and add batch dimension
    img_trans = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # Get model's prediction
    prediction_tensor = model(img_trans)
    
    # Get the predicted class index (0 or 1)
    class_index = torch.argmax(prediction_tensor).item()
    
    # Convert class index back to label (NORMAL or PNEUMONIA)
    final_prediction = label_encoder.inverse_transform([class_index])[0]
    
    return final_prediction

# Path to test image
img_path = r"data\resize_image\test_NORMAL_IM-0006-0001.jpeg"

# Load the image
img = Image.open(img_path)

# Make prediction on the image
prediction = infer(img)
print(prediction)

