import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import random
from PIL import Image
import os

# --- Set this flag: TRAIN = True for training, TRAIN = False for loading ---
TRAIN = False  # Set to True if you want to train and save, False to load and skip training

MODEL_PATH = "resunitnet_glioma.pth"

# --- Residual Unit Definition ---
class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# --- ResUnit-based Classifier ---
class ResUnitNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ResUnitNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = ResUnit(32, 32)
        self.layer2 = ResUnit(32, 64, stride=2)
        self.layer3 = ResUnit(64, 128, stride=2)
        self.layer4 = ResUnit(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

# --- Data preparation ---
train_dir = r'C:\Users\Suchi\Downloads\archive\Training'
test_dir = r'C:\Users\Suchi\Downloads\archive\Testing'

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
class_names = train_data.classes

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUnitNet(num_classes=len(class_names)).to(device)

# --- Training or Loading ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_losses, test_losses = [], []
train_correct, test_correct = [], []

if TRAIN or not os.path.exists(MODEL_PATH):
    print("Training the model...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_losses.append(running_loss / total)
        train_correct.append(correct)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        test_losses.append(running_loss / total)
        test_correct.append(correct)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {test_losses[-1]:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

else:
    print("Loading the model from file...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

# --- Plotting Loss & Accuracy (only if training just happened) ---
if TRAIN:
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), test_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.show()

    train_acc = [c / len(train_data) for c in train_correct]
    val_acc = [c / len(test_data) for c in test_correct]
    plt.plot(range(epochs), train_acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy per Epoch')
    plt.show()

# --- Final test accuracy and confusion matrix ---
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds

test_accuracy, all_labels, all_preds = calculate_accuracy(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.2f}%')

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- Visualize Correct and Incorrect Predictions ---
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0, 0, 0])  # Change if you normalized your images
    std = np.array([1, 1, 1])   # Change if you normalized your images
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Get one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Identify correct and incorrect predictions
correct_indices = (preds == labels).cpu().numpy().nonzero()[0]
incorrect_indices = (preds != labels).cpu().numpy().nonzero()[0]

# Randomly sample up to 5 correct and 5 incorrect predictions
num_samples = min(5, len(correct_indices), len(incorrect_indices), 5)
if num_samples > 0:
    sampled_correct = np.random.choice(correct_indices, num_samples, replace=False)
    sampled_incorrect = np.random.choice(incorrect_indices, num_samples, replace=False)

    plt.figure(figsize=(15, 6))

    # Plot correct predictions
    for i, idx in enumerate(sampled_correct):
        plt.subplot(2, num_samples, i+1)
        imshow(images[idx])
        plt.title(f"Correct\nTrue: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")

    # Plot incorrect predictions
    for i, idx in enumerate(sampled_incorrect):
        plt.subplot(2, num_samples, num_samples+i+1)
        imshow(images[idx])
        plt.title(f"Incorrect\nTrue: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")

    plt.tight_layout()
    plt.show()
else:
    print("Not enough correct/incorrect samples to display.")

# --- Inference for a New Image ---
def predict_image(image_path, model, class_names, device):
    image_path = r'C:\Users\Suchi\Pictures\glioma1.jpg'
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img_t = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(img_t)
        _, pred = torch.max(out, 1)
    predicted_class = class_names[pred.item()]
    return predicted_class

# Example usage:
# image_path = r'C:\path\to\your\new_image.jpg'
# prediction = predict_image(image_path, model, class_names, device)
# print(f'The model predicts: {prediction}')
# img = Image.open(image_path)
# plt.imshow(img)
# plt.title(f'Predicted: {prediction}')
# plt.axis('off')

# plt.show()