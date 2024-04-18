import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn

BATCH_SIZE = 16
NUM_CLASSES = 5

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6521, 0.5233, 0.5159], std=[0.2284, 0.2071, 0.2186])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6521, 0.5233, 0.5159], std=[0.2284, 0.2071, 0.2186])
    ])
}

data_dir = 'isic2019_modified'
val_dir = f'{data_dir}/val'
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val']) 
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dir = f'{data_dir}/test'
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
effnets = [getattr(models, f'efficientnet_b{d}')() for d in range(8)]
for i in range(8):
    model = effnets[i]
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model_path = 'normalised_and_without_augments/base_b' + str(i) + '.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

def ensemble_predict(inputs):
    predictions = []
    for model in effnets:
        outputs = model(inputs)
        predictions.append(outputs) 
    ensemble_predictions = torch.stack(predictions) 
    # predictions is of size (8, 16, 5)
    model_votes = torch.argmax(ensemble_predictions, dim=-1)  
    majority_votes, _ = torch.mode(model_votes, dim=0)
    return majority_votes  

### VAL SET ###
all_preds = []
all_labels = []

# No gradient calculation needed
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = ensemble_predict(images) 
        
        all_preds.extend(predicted.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate weighted accuracy
class_counts = np.bincount(all_labels)
class_accuracy = [accuracy_score(all_labels[all_labels == cls], all_preds[all_labels == cls]) for cls in range(len(class_counts))]
print(class_accuracy)
weighted_accuracy = np.average(class_accuracy, weights=class_counts / len(all_labels))
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print the metrics
print(f'Accuracy on the val set: {weighted_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(cm)

print(classification_report(all_labels, all_preds))

### TEST SET ###
all_preds = []
all_labels = []

# No gradient calculation needed
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = ensemble_predict(images)  # Use your ensemble prediction function
        
        all_preds.extend(predicted.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate weighted accuracy
class_counts = np.bincount(all_labels)
class_accuracy = [accuracy_score(all_labels[all_labels == cls], all_preds[all_labels == cls]) for cls in range(len(class_counts))]
print(class_accuracy)
weighted_accuracy = np.average(class_accuracy, weights=class_counts / len(all_labels))
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print the metrics
print(f'Accuracy on the test set: {weighted_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

cm = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(cm)

print(classification_report(all_labels, all_preds))


