import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.init as init
import os
import pickle
from typing import List
import cv2

category_mapping = {
    'r_set': 0,
    "r_winpoint": 1,
    'r-pass': 2,
    'r_spike': 3,
    'l_set': 4,
    'l_winpoint': 5,
    'l-pass': 6,
    'l-spike': 7
}

class VolleyballDataset(Dataset):
    def __init__(self, videos_annot, transform=None):
        self.videos_annot = videos_annot
        self.transform = transform
        self.data = []
        self.frames = []
        self.indices = []
        for video in videos_annot:
            for clip in videos_annot[video]:
                category_idx = category_mapping.get(videos_annot[video][clip]['category'])
                self.frames.append((f"{dataset_root}/videos/{video}/{clip}/{clip}.jpg", category_idx, int(clip)))
                self.indices.append(int(clip))
                for frame in videos_annot[video][clip]['frame_boxes_dct']:
                    frame_path = f"{dataset_root}/videos/{video}/{clip}/{frame}.jpg"
                    if os.path.exists(frame_path):
                        self.data.append((frame_path, videos_annot[video][clip]['category'], videos_annot[video][clip]['frame_boxes_dct'][frame]))
    
    def __len__(self):
        return len(self.frames)
    
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        # img_path, category, boxes = self.data[idx]
        # image = Image.open(img_path).convert('RGB')    for each frame in clip
        # category_idx = category_mapping.get(category)
        
        
        img_path, category,clip = self.frames[idx] # for each clip only
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        
        category_one_hot = torch.nn.functional.one_hot(torch.tensor(category), num_classes=len(category_mapping))
        
        return image, category_one_hot ,original_idx

def load_volleyball_dataset(dataset_root):
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)
    return videos_annot

def get_split_indices(videos_annot, train_videos, val_videos, test_videos):
    train_indices = []
    val_indices = []
    test_indices = []
    for video in videos_annot:
        for clip in videos_annot[video]:
            if int(video) in train_videos:
                train_indices.append(int(clip))
            elif int(video) in val_videos:
                val_indices.append(int(clip))
            elif int(video) in test_videos:
                test_indices.append(int(clip))
    return train_indices, val_indices, test_indices

dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"
videos_annot = load_volleyball_dataset(dataset_root)

# Indices for each split
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_indices, val_indices, test_indices = get_split_indices(videos_annot, train_videos, val_videos, test_videos)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def subset(full_dataset, indices):
    subset_indices = [full_dataset.indices.index(i) for i in indices if i in full_dataset.indices]
    return Subset(full_dataset, subset_indices)

full_dataset = VolleyballDataset(videos_annot, transform=transform)



train_dataset = subset(full_dataset, train_indices)
val_dataset = subset(full_dataset, val_indices)
test_dataset = subset(full_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)






# Load the pre-trained ResNet-152 model
model = models.resnet152(pretrained=True)

# Freeze all the layers
# for param in model.parameters():
#     param.requires_grad = False

# Get the number of classes
num_classes = len(category_mapping)




def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

model.fc = nn.Sequential(
    nn.Identity(),       # Remove the original fully connected layer
    nn.Linear(2048, 2024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(2024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)

# Apply the weight initialization
# model.fc.apply(initialize_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Variables to track the best model
best_val_accuracy = 0.0
best_model_weights = None

num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, clips in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, clips in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")
    
    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()

# Save the best model weights
torch.save(best_model_weights, 'modified_rsnt_model.pth')

print('Finished Training')
print(f'Best Validation Accuracy: {best_val_accuracy}')

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels, clips in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.argmax(dim=1)).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# test acc = 76
