#Loading the MNIST MODEL
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn as nn
import numpy as np
import numpy as np

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = MNISTNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
model.eval()
print("Model loaded and ready for inference!")

# Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_mnist_probabilities(image_path: str) -> str:
    try:
        # Load and convert image to grayscale
        image = Image.open(image_path).convert("L")
        
        # Check if image needs inversion 
        # MNIST has BLACK background with WHITE digits
        img_array = np.array(image)
        
        # If the average pixel value is HIGH (white/light background), invert the image
        # to match MNIST format (black background, white digits)
        if np.mean(img_array) > 127:
            image = ImageOps.invert(image)
        
        # Apply the same transformations used during training
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Get predictions
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        # Create a string representation
        result = "\n".join([f"{i}: {prob:.4f}" for i, prob in enumerate(probabilities)])
        return result
    
    except Exception as e:
        return f"Error: {e}"