import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

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
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = DeviceClassifier(num_classes=4)
    model.load_state_dict(torch.load('device_model.pth'))
    model.eval()

    from PIL import Image
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    classes = ['mobile', 'laptop', 'battery', 'charger']
    return classes[torch.argmax(output).item()]

if __name__ == '__main__':
    train_model()