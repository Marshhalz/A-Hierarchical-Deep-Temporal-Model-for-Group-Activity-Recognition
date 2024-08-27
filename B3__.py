import os
import pickle
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.optim as optim
import random
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from torchsummary import summary






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




model = models.resnet152(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.fc = nn.Sequential(
    nn.Identity(),       
    nn.Linear(2048, 2024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(2024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 8)
)

model.load_state_dict(torch.load(r'Z:\work\cnn\volley_project\modified_rsnt_model.pth'))


children_list = list(model.children())
model = nn.Sequential(*(children_list[:-1]))

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
    def __init__(self, videos_annot, video_root, model, transform=None):
        self.videos_annot = videos_annot
        self.transform = transform
        self.data = []
        self.indices = []
        
        videos_dirs = os.listdir(video_root)
        videos_dirs.sort()
        
        for idx, video_dir in enumerate(videos_dirs):
            video_dir_path = os.path.join(video_root, video_dir)
            if not os.path.isdir(video_dir_path):
                continue
            clips_dir = os.listdir(video_dir_path)
            clips_dir.sort()
            
            for clip_dir in clips_dir:
                clip_dir_path = os.path.join(video_dir_path, clip_dir)
                if not os.path.isdir(clip_dir_path):
                    continue

                annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
                frame_boxes = load_tracking_annot(annot_file)

                with torch.no_grad():
                    preprocessed_images = []
                    for frame_id, box_info in frame_boxes.items():
                        print(frame_id)
                        img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                        image = Image.open(img_path).convert('RGB')
                        
                        if self.transform:
                            image = self.transform(image)
                        preprocessed_images.append(image.unsqueeze(0))
                    
                    preprocessed_images = torch.cat(preprocessed_images)
                    preprocessed_images = preprocessed_images.to(device)
                    model = model.to(device)  

                    dnn_repr = model(preprocessed_images)
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1)
                    
                    category_idx = category_mapping.get(videos_annot[video_dir][clip_dir]['category'])
                    self.indices.append(int(clip_dir))
                    self.data.append((dnn_repr, category_idx, int(clip_dir)))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        datav, category, clip = self.data[idx]
        category_one_hot = torch.nn.functional.one_hot(torch.tensor(category), num_classes=len(category_mapping))
        return datav, category_one_hot, original_idx

def load_volleyball_dataset(dataset_root):
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)
    return videos_annot

videos_annot = load_volleyball_dataset(dataset_root)

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# full_dataset = VolleyballDataset(videos_annot, video_root, model, transform)
# torch.save(full_dataset, 'full_dataset_on_hole_image_9_frames.pth')




full_dataset = torch.load('full_dataset_on_hole_image_9_frames.pth')



train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_indices, val_indices, test_indices = get_split_indices(videos_annot, train_videos, val_videos, test_videos)
num_classes = len(category_mapping)

def subset(full_dataset, indices):
    subset_indices = [full_dataset.indices.index(i) for i in indices if i in full_dataset.indices]
    return Subset(full_dataset, subset_indices)

train_dataset = subset(full_dataset, train_indices)
val_dataset = subset(full_dataset, val_indices)
test_dataset = subset(full_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)






class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define parameters
input_size = 2048
hidden_size = 126
num_layers = 1
num_classes = len(category_mapping)
num_epochs = 100
batch_size = 16
learning_rate = 0.0001


model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.1)


best_val_accuracy = 0.0
best_model_path = 'best_lstm_model.pth'


for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    for data, targets, _ in train_loader:
        data = data.to(device)
        targets = torch.argmax(targets, dim=1).to(device)
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Validation loop
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, targets, _ in val_loader:
            data = data.to(device)
            targets = torch.argmax(targets, dim=1).to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            

            _, predicted = torch.max(outputs.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
            
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved new best model with Validation Accuracy: {best_val_accuracy:.2f}%')

# Test loop
model.eval()
test_loss = 0
correct_test = 0
total_test = 0
with torch.no_grad():
    for data, targets, _ in test_loader:
        data = data.to(device)
        targets = torch.argmax(targets, dim=1).to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_test += targets.size(0)
        correct_test += (predicted == targets).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct_test / total_test
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')



#75