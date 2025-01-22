

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image
import os

from google.colab import drive
drive.mount('/content/drive')

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(image_dir):
            class_folder = os.path.join(image_dir, label)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(class_folder, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = '/content/drive/MyDrive/task-1/split_dataset/train'
val_dir = '/content/drive/MyDrive/task-1/split_dataset/val'
test_dir = '/content/drive/MyDrive/task-1/split_dataset/test'

train_dataset = CustomDataset(train_dir, transform=data_transforms)
val_dataset = CustomDataset(val_dir, transform=data_transforms)
test_dataset = CustomDataset(test_dir, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model setup complete!")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Loss function and optimizer setup complete!")

import time

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()

    for inputs, labels in train_loader:
        step_start_time = time.time()

        inputs = inputs.to(device)
        labels = torch.tensor([label_map[label] for label in labels]).to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        print(f"Step time: {step_time:.4f} seconds")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Epoch time: {epoch_time:.2f} seconds")

model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = torch.tensor([label_map[label] for label in labels]).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = torch.tensor([label_map[label] for label in labels]).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
cm = confusion_matrix(all_labels, all_preds)
class_names = list(label_map.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(all_labels, all_preds, target_names=class_names))

torch.save(model.state_dict(), '/content/drive/MyDrive/eye_disease_modelMobileNetV2.pth')
print("Model saved successfully!")

def predict_image(image_path, model, label_map):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = list(label_map.keys())[list(label_map.values()).index(predicted.item())]
    return predicted_class

predicted_class = predict_image('/content/drive/MyDrive/task-1/split_dataset/test/cataract/2109_right.jpg', model, label_map)
print(f"Predicted Class: {predicted_class}")

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

label_map = {'cataract': 0, 'diabetic_retinopathy': 1, 'glaucoma': 2, 'normal': 3}
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load('/content/drive/MyDrive/eye_disease_modelMobileNetV2.pth', weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path, model, label_map):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = list(label_map.keys())[list(label_map.values()).index(predicted_class_idx.item())]
        confidence = confidence.item() * 100
    return predicted_class, confidence, image

image_paths = [
    '/content/drive/MyDrive/task-1/split_dataset/test/cataract/2108_left.jpg',
    '/content/drive/MyDrive/task-1/split_dataset/test/diabetic_retinopathy/10085_left.jpeg',
    '/content/drive/MyDrive/task-1/split_dataset/test/glaucoma/_105_9159338.jpg',
    '/content/drive/MyDrive/task-1/split_dataset/test/normal/2382_right.jpg'
]

for image_path in image_paths:
    predicted_class, confidence, image = predict_image(image_path, model, label_map)
    print(f"Predicted Class: {predicted_class} with Confidence: {confidence:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()
    