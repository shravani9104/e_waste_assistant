import os
import requests
import json

def import_torch_modules():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
    return torch, nn, optim, datasets, transforms, models, DataLoader

# Define CNN model
class DeviceClassifier:
    def __init__(self, num_classes=4):
        torch, nn, optim, datasets, transforms, models, DataLoader = import_torch_modules()
        class _DeviceClassifier(nn.Module):
            def __init__(self, num_classes):
                super(_DeviceClassifier, self).__init__()
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            def forward(self, x):
                return self.model(x)

        self.model = _DeviceClassifier(num_classes)

    def __call__(self, x):
        return self.model(x)

# Training function
def train_model():
    torch, nn, optim, datasets, transforms, models, DataLoader = import_torch_modules()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder('data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = DeviceClassifier(num_classes=4).model
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
        torch, nn, optim, datasets, transforms, models, DataLoader = import_torch_modules()
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Check if model file exists
        if not os.path.exists('device_model.pth'):
            print("Model file 'device_model.pth' not found in project root.")
            print("Please train the model by running `python model.py` to generate it or add a pretrained model file.")
            print("Falling back to filename-based device classification.")
            return classify_by_filename(image_path)

        model = DeviceClassifier(num_classes=4).model
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

def call_gemini_api(device_type, age, condition, damage, working_parts, buying_price):
    """
    Calls Gemini API model to estimate recycling value based on device info.
    This is a placeholder implementation. Replace with real API call.
    """
    try:
        api_url = "https://api.gemini.example.com/estimate"  # Replace with actual Gemini API URL
        api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")

        payload = {
            "device_name": device_type,
            "age": age,
            "working_state": condition,
            "working_parts": working_parts,
            "buying_price": buying_price
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Assume the response has a field 'estimated_value' with the value string
            estimated_value = data.get('estimated_value', 'No estimate available')
            return estimated_value
        else:
            print(f"Gemini API error: {response.status_code} {response.text}")
            return "Unable to get estimate at this time."
    except Exception as e:
        print(f"Exception calling Gemini API: {e}")
        return "Error occurred while estimating value."

if __name__ == '__main__':
    train_model()
