import os
import pickle
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import random
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F

from PIL import __version__ as PILLOW_VERSION


annot_root= r"Z:\work\cnn\volley_project\data_set\volleyball_tracking_annotation\volleyball_tracking_annotation\_"

video_root=r"Z:\work\cnn\volley_project\data_set\volleyball_\videos"


dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        
        self.player_ID = words[0]
        del words[0]
        
        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated







def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx:[] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            #9 frames only
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct


def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f'{frame_id}.jpg')
        image = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

        cv2.imshow('Image', image)
        cv2.waitKey(180)
    cv2.destroyAllWindows()



def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        # print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            #print(f'\t{clip_dir_path}')
            assert clip_dir in clip_category_dct

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            frame_boxes_dct = load_tracking_annot(annot_file)
            #vis_clip(annot_file, clip_dir_path)

            clip_annot[clip_dir] = {
                'category': clip_category_dct[clip_dir],
                'frame_boxes_dct': frame_boxes_dct
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    annot_file= r"Z:\work\cnn\volley_project\data_set\volleyball_tracking_annotation\volleyball_tracking_annotation\_"
    video_dir=r"Z:\work\cnn\volley_project\data_set\volleyball_\videos"

    videos_annot = load_volleyball_dataset(video_dir, annot_file)

    with open(f'{dataset_root}/annot_all.pkl', 'wb') as file:
        pickle.dump(videos_annot, file)

# if __name__ == '__main__':

#     # vis_clip(annot_file, clip_dir_path)
    
#     create_pkl_version()











max_pooled_features = os.listdir(r"Z:\work\cnn\volley_project\data_set\volleyball_\features_maxed\image-level\resnet")
max_pooled_features.sort()






# Category mapping
category_mapping = {
    'r_set': 0,
    'r_winpoint': 1,
    'r-pass': 2,
    'r_spike': 3,
    'l_set': 4,
    'l_winpoint': 5,
    'l-pass': 6,
    'l-spike': 7
}


class VolleyballDataset(Dataset):
    def __init__(self, max_pooled_features, videos_annot):
        self.feature = []
        self.data = []
        self.frames = []
        self.indices = []
        
        for video in max_pooled_features:
            for clip in videos_annot[video]:
                category_idx = category_mapping.get(videos_annot[video][clip]['category'])
                
                if category_idx is None:
                    continue  # Skip this entry if the category is not in the mapping
                
                feature_path = os.path.join(r'Z:\work\cnn\volley_project\data_set\volleyball_\features_maxed\image-level\resnet', video, f'{clip}.npy')
                self.feature.append((feature_path, category_idx, int(clip)))
                self.indices.append(int(clip))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        
        f_path, category, clip = self.feature[idx]
        loaded_features = np.load(f_path)
        
        category_one_hot = torch.nn.functional.one_hot(torch.tensor(category), num_classes=len(category_mapping))
        
        return torch.tensor(loaded_features, dtype=torch.float32), category_one_hot.float(), original_idx


def load_volleyball_dataset(dataset_root):
    with open(os.path.join(dataset_root, 'annot_all.pkl'), 'rb') as file:
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


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        

        loss.backward()
        optimizer.step()
        

        running_loss += loss.item() * inputs.size(0)
        

        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        corrects += torch.sum(preds == labels)
        total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = corrects.double() / total
    
    return epoch_loss, accuracy


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            

            running_loss += loss.item() * inputs.size(0)
            

            _, preds = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            corrects += torch.sum(preds == labels)
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = corrects.double() / total
    
    return epoch_loss, accuracy


videos_annot = load_volleyball_dataset(dataset_root)


train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_indices, val_indices, test_indices = get_split_indices(videos_annot, train_videos, val_videos, test_videos)


full_dataset = VolleyballDataset(max_pooled_features, videos_annot)

train_dataset = Subset(full_dataset, [full_dataset.indices.index(i) for i in train_indices])
val_dataset = Subset(full_dataset, [full_dataset.indices.index(i) for i in val_indices])
test_dataset = Subset(full_dataset, [full_dataset.indices.index(i) for i in test_indices])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)







class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.02))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# input_dim = 2048
hidden_dims = [2048,4048,2048,4048,2048,2048,1024, 512, 512]
output_dim = len(category_mapping)  
num_epochs =  250


model = MLP(input_dim=2048, hidden_dims=hidden_dims, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_val_accuracy = 0.0
best_model_path = 'v2-best_B2_model.pth'


for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with validation accuracy: {val_accuracy:.4f}')
        
        
        
        



model.load_state_dict(torch.load(r"Z:\work\cnn\volley_project\v2-best_B2_model.pth"))


model.to(device)


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

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


#  65



