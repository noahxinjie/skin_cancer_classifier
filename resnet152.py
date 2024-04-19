# Import necessary packages
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the directory for train and validation datasets
data_dir = 'data'
train_dir = f'{data_dir}/train'
val_dir = f'{data_dir}/val'
test_dir = f'{data_dir}/test'

# Parameters
num_classes = 5  # Change this based on your dataset
batch_size = 32
num_epochs = 30

# Define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.6521, 0.5233, 0.5159], [0.2284, 0.2071, 0.2186])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.6521, 0.5233, 0.5159], [0.2284, 0.2071, 0.2186])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.6521, 0.5233, 0.5159], [0.2284, 0.2071, 0.2186])
    ])
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test']),
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4),
}

# Load a pre-trained model and modify it for your number of classes
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)
CLASS_WEIGHTS = [2.1565526541620694, 4.945459657014338, 4.322113022113022, 30.173241852487134, 14.37173202614379]
# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(CLASS_WEIGHTS).cuda())
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloaders["train"])}')

    # After each epoch, evaluate accuracy on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = correct / total
    print(f'Validation Accuracy after epoch {epoch+1}: {epoch_acc:.4f}')

    # Save the model if it has a better accuracy than the best model seen so far
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), 'best_model_trans.pth')
        print('Model saved as best_model.pth')

print('Finished Training')

# Load the model for evaluation
#model = models.resnet50(pretrained=False)  # Initialize the model architecture
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust the final layer as before

#model.load_state_dict(torch.load('best_model.pth'))
#model = model.to(device)
#model.eval()  # Set the model to evaluation mode

# Now you can use the model for evaluation or further training

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

model.load_state_dict(torch.load('best_model_trans.pth'))
model = model.to(device)
# Ensure the model is in evaluation mode
model.eval()

# Lists to store all predictions and true labels
all_preds = []
all_labels = []

# No gradient calculation needed
with torch.no_grad():
    for images, labels in dataloaders['test']:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Get the predictions
        _, predicted = torch.max(outputs.data, 1)
        
        # Append current predictions and labels to the lists
        all_preds.extend(predicted.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

# Convert lists to numpy arrays for evaluation metrics calculation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = np.sum(all_preds == all_labels) / len(all_labels)

# Calculate precision, recall, and F1 score
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# Print the metrics
print(f'Accuracy on the test set: {accuracy:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# If you're interested in the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(cm)


