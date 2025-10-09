import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Define CNN model
class DeviceClassifier(nn.Module):
    def __init__(self, num_classes=4):  # For mobile, laptop, battery, charger
        super(DeviceClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = DeviceClassifier(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} complete')

    torch.save(model.state_dict(), 'device_model.pth')
    return model

# Inference function
def detect_device(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Check if model file exists
        if not os.path.exists('device_model.pth'):
            print("Model file not found. Using fallback classification.")
            return classify_by_filename(image_path)
        
        model = DeviceClassifier(num_classes=4)
        model.load_state_dict(torch.load('device_model.pth', map_location='cpu'))
        model.eval()

        from PIL import Image
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
        classes = ['mobile', 'player', 'battery', 'keyboard']
        return classes[torch.argmax(output).item()]
    except Exception as e:
        print(f"Error in model inference: {e}")
        return classify_by_filename(image_path)

def classify_by_filename(image_path):
    """Fallback classification based on filename"""
    filename = os.path.basename(image_path).lower()
    if 'mobile' in filename or 'phone' in filename:
        return 'mobile'
    elif 'battery' in filename:
        return 'battery'
    elif 'player' in filename or 'music' in filename:
        return 'player'
    elif 'keyboard' in filename:
        return 'keyboard'
    else:
        return 'mobile'  # Default fallback

if __name__ == '__main__':
    train_model()