import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.data import DataLoader, random_split

data_dir = 'isic2019_modified'
train_dir = f'{data_dir}/train'
val_dir = f'{data_dir}/val'

# Parameters
NUM_CLASSES = 5 
BATCH_SIZE = 32
NUM_EPOCHS = 25
CLASS_WEIGHTS = {0: 2.1565526541620694, 1: 4.945459657014338, 2: 4.322113022113022, 3: 30.173241852487134, 4: 14.37173202614379}


# Define transformations
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
"""
mean = [0.6521, 0.5233, 0.5159]
std = [0.2284, 0.2071, 0.2186]
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


"""
full_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True),
}

"""

image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val']),
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,),
}


from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch

# Load a pre-trained EfficientNet model and modify it for your number of classes
model = models.efficientnet_b0(pretrained=True)  # You can change this to b1, b2, etc., based on your needs

class DropoutEfficentNet(nn.Module):
    def __init__(self):
        super(DropoutEfficentNet, self).__init__()
        self.features = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.features.classifier[1].in_features
        self.features.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return x
    
model = DropoutEfficentNet()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

# Loss and optimizer
class_weights_list = list(CLASS_WEIGHTS.values())
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

best_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_epoch_loss = running_loss/len(dataloaders["train"])
    train_epoch_acc = correct / total

    print(f'Epoch {epoch+1}, Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_acc:.4f}')

    # After each epoch, evaluate accuracy on the validation set
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).to(device)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = correct / total
    print(f'Epoch {epoch+1}: Val loss: {val_loss:.4f}, Val accuracy: {epoch_acc:.4f}')

    scheduler.step(val_loss)
    torch.save(model.state_dict(), 'dropout_b0.pth')
    print('Model saved')
print('Finished Training')
