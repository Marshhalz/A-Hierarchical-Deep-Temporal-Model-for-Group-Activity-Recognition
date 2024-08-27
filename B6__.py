import os
import pickle
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, Subset,TensorDataset
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




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                # Add the box_info to the corresponding frame
                frame_boxes_dct[box_info.frame_ID].append(box_info)

        # Sort each list of boxes by the x-axis
        for frame_ID in frame_boxes_dct:
            frame_boxes_dct[frame_ID].sort(key=lambda box: box.x_axis)

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

    with open(f'{dataset_root}/annot_all_pl_sorted.pkl', 'wb') as file:
        pickle.dump(videos_annot, file)

# if __name__ == '__main__':

#     # vis_clip(annot_file, clip_dir_path)
    
    # create_pkl_version()


category_mapping = {
    'waiting': 0,
    "setting": 1,
    'digging': 2,
    'falling': 3,
    'spiking': 4,
    'blocking': 5,
    'jumping': 6,
    'moving': 7,
    'standing': 8
}


category_mapping1 = {
    'r_set': 0,
    "r_winpoint": 1,
    'r-pass': 2,
    'r_spike': 3,
    'l_set': 4,
    'l_winpoint': 5,
    'l-pass': 6,
    'l-spike': 7
}

def load_volleyball_dataset(dataset_root):
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)
    return videos_annot


dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"
videos_annot = load_volleyball_dataset(dataset_root)

def load_feature_files(video_ids,videos_annot, feature_root):
    data = []
    for video_id in video_ids:
        video_dir = os.path.join(feature_root, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        clips = os.listdir(video_dir)
        clips.sort()

        for clip in clips:
            clip_path = os.path.join(video_dir, clip)
            if clip_path.endswith('.npy'):
                features = np.load(clip_path, allow_pickle=True).item()
                category = videos_annot[str(video_id)][str(clip[:-4])]['category']
                category_idx = category_mapping1.get(category)
                data.append((features,category_idx))
    return data

feature_root = r'Z:\work\cnn\volley_project\data_set\volleyball_\features_p\player-level\resnet'
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_data = load_feature_files(train_videos,videos_annot, feature_root)
val_data = load_feature_files(val_videos,videos_annot, feature_root)
test_data = load_feature_files(test_videos,videos_annot, feature_root)



class VolleyballDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sequences = []
        self.labels = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        for features,_ in self.data:
            for player_id, (pid, dnn_repr_list, category) in features.items():
                if len(dnn_repr_list) < 9:
                    continue  # Skip if less than 9 frames

                for i in range(0, len(dnn_repr_list) - 9 + 1):
                    sequence = np.array(dnn_repr_list[i:i+9])
                    self.sequences.append(sequence)
                    self.labels.append(category_mapping[category])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = VolleyballDataset(train_data)
val_dataset = VolleyballDataset(val_data)
test_dataset = VolleyballDataset(test_data)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Final output for classification from the last time step
        final_output = self.fc(out[:, -1, :])
        
        # Return the final output and all hidden states
        return final_output, out



input_size = 2048  # Based on the output of ResNet
hidden_size = 1024
num_layers = 1
num_classes = len(category_mapping)

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)




num_epochs = 20
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for sequences, labels in train_loader:
        # Ensure the correct shape
        batch_size, seq_len, num_players, feature_dim1, feature_dim2, feature_dim3 = sequences.shape
        sequences = sequences.view(batch_size, seq_len, num_players * feature_dim1 * feature_dim2 * feature_dim3)
        
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs,all_hidden_states = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            batch_size, seq_len, num_players, feature_dim1, feature_dim2, feature_dim3 = sequences.shape
            sequences = sequences.view(batch_size, seq_len, num_players * feature_dim1 * feature_dim2 * feature_dim3)
            
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs,all_hidden_states = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')


    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"Model saved with Val Accuracy: {val_accuracy:.2f}%")







model.eval()


test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for sequences, labels in test_loader:

        batch_size, seq_len, num_players, feature_dim1, feature_dim2, feature_dim3 = sequences.shape
        sequences = sequences.view(batch_size, seq_len, num_players * feature_dim1 * feature_dim2 * feature_dim3)
        
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs,all_hidden_states = model(sequences)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')





def load_feature(dataset, model):
    model.eval()  # Set the model to evaluation mode
    
    all_clip_features = []
    
    with torch.no_grad():
        for x in range(len(dataset)):
            hidden_states = []
            # Debugging: Print the structure of dataset[x]
            # print(f"Clip {x}: {dataset[x]}")
            
            for i in range(len(dataset[x][0])):  # Iterate over the 12 players
                try:
                    pid, dnn_repr_list, category = dataset[x][0][i]
                except TypeError as e:
                    # print(f"Error at dataset[{x}][{i}]: {e}")
                    continue

                if len(dnn_repr_list) < 9:
                    continue  # Skip if less than 9 frames
                
                # Prepare the input tensor
                x_tensor = torch.tensor(dnn_repr_list, dtype=torch.float32)
                seq_len, num_players, feature_dim1, feature_dim2, feature_dim3 = x_tensor.shape
                x_tensor = x_tensor.view(seq_len, num_players * feature_dim1 * feature_dim2 * feature_dim3)
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension
                x_tensor = x_tensor.to(device)
                
                # Forward pass through the model
                outputs, all_hidden_states = model(x_tensor)
                
                # Remove batch dimension to get (9, hidden_size)
                all_hidden_states = all_hidden_states.squeeze(0)
                hidden_states.append(all_hidden_states)  # Append hidden states of the player
            
            if hidden_states:  # Check if there are any hidden states to process
                # Stack hidden states to get a tensor of shape (12, 9, hidden_size)
                hidden_states = torch.stack(hidden_states)
                
                
                max_pooled_hidden_states = torch.max(hidden_states, dim=0)[0]  # Result shape: (9, hidden_size)
                
                
                all_clip_features.append((max_pooled_hidden_states,dataset[x][1]))
    
    return all_clip_features



train_f2  = load_feature(train_data, model)
val_f2 = load_feature(val_data, model)
test_f2 = load_feature(test_data, model)







def create_dataloader(features, batch_size=32):
    feature_list, labels = zip(*features)
    
    feature_tensor = torch.stack(feature_list)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(feature_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


train_loader = create_dataloader(train_f2)
val_loader = create_dataloader(val_f2)
test_loader = create_dataloader(test_f2)





class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = 1024
hidden_size = 128 
num_layers = 2
num_classes = 8 

model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)




def train_model(model, train_loader, val_loader, num_epochs=300):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=0.00001)
    # optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-9)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = val_correct / val_total
        print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}')


train_model(model, train_loader, val_loader)


def test_model(model, test_loader):
    model.eval()  
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad(): 
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_accuracy = correct / total

    print(f'Test Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_accuracy:.4f}')
    return test_epoch_loss, test_epoch_accuracy


test_loss, test_accuracy = test_model(model, test_loader)






#60
