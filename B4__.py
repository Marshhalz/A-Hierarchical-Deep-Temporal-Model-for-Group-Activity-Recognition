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




def prepare_model():

    model = models.resnet152(pretrained=True)
    
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_classes = len(category_mapping)
    model.fc = nn.Sequential(
        nn.Identity(),       # Remove the original fully connected layer
        nn.Linear(2048, 2024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(2024, 2048),
        nn.ReLU(),
        nn.Linear(2048, num_classes))
    
    model.load_state_dict(torch.load(r'Z:\work\cnn\volley_project\vanila_rsnt_model_on_players.pth'))
    
    
    model.to(device)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    return model, preprocess









def extract_features(clip_dir_path, annot_file, output_file, model, preprocess):
    frame_boxes = load_tracking_annot(annot_file)
    
    per_player={}
    cat_pl={}
    
    data={}

    with torch.no_grad():
        for frame_id, boxes_info in frame_boxes.items():
            try:
                img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')
                for box_info in boxes_info:
                    if box_info.player_ID not in per_player:
                        per_player[box_info.player_ID] = []
                    x1, y1, x2, y2 = box_info.box
                    cropped_image = image.crop((x1, y1, x2, y2))
                    cat_pl[box_info.player_ID] = box_info.category
                    preprocessed_image = preprocess(cropped_image).unsqueeze(0).to(device)
                    dnn_repr = model(preprocessed_image)
                    dnn_repr = dnn_repr.cpu().numpy()
                    per_player[box_info.player_ID].append(dnn_repr)
                
                for p_id, dnn_repr_list in per_player.items():
                    # print(p_id)
                    data[p_id] = (p_id, dnn_repr_list, cat_pl[p_id])
                np.save(output_file, data)
            except Exception as e:
                print(f"An error occurred: {e}")




annot_root= r"Z:\work\cnn\volley_project\data_set\volleyball_tracking_annotation\volleyball_tracking_annotation\_"

video_root=r"Z:\work\cnn\volley_project\data_set\volleyball_\videos"


dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"



if __name__ == '__main__':


    model, preprocess = prepare_model()

    output_root = f'{dataset_root}/features_p/player-level/resnet'

    videos_dirs = os.listdir(video_root)
    videos_dirs.sort()

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(video_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue

            print(f'\t{clip_dir_path}')

            annot_file = os.path.join(annot_root, video_dir, clip_dir, f'{clip_dir}.txt')
            output_file = os.path.join(output_root, video_dir)

            if not os.path.exists(output_file):
                os.makedirs(output_file)

            output_file = os.path.join(output_file, f'{clip_dir}.npy')
            extract_features(clip_dir_path, annot_file, output_file, model, preprocess)






def load_feature_files(video_ids, feature_root):
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
                data.append(features)
    return data

feature_root = r'Z:\work\cnn\volley_project\data_set\volleyball_\features_p\player-level\resnet'
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_data = load_feature_files(train_videos, feature_root)
val_data = load_feature_files(val_videos, feature_root)
test_data = load_feature_files(test_videos, feature_root)




class VolleyballDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sequences = []
        self.labels = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        for features in self.data:
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





class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 2048  # Based on the output of ResNet
hidden_size = 1024
num_layers = 1
num_classes = len(category_mapping)

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)









num_epochs = 20
best_val_accuracy = 0.0
best_model_path = "bestlstm_b4_on_pl_model_v2.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for sequences, labels in train_loader:
        # Ensure the correct shape
        batch_size, seq_len, num_players, feature_dim1, feature_dim2, feature_dim3 = sequences.shape
        sequences = sequences.view(batch_size, seq_len, num_players * feature_dim1 * feature_dim2 * feature_dim3)
        
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
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

            outputs = model(sequences)
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
        torch.save(model.state_dict(), best_model_path)
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

        outputs = model(sequences)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')




class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]  




input_size = 2048  
hidden_size = 1024
num_layers = 1
feature_extractor = LSTMFeatureExtractor(input_size, hidden_size, num_layers).to(device)

# Load the pretrained weights
best_model_path = "bestlstm_b4_on_pl_model_v2.pth"
state_dict = torch.load(best_model_path, map_location=device)

# Only load the LSTM weights from the state dictionary
feature_extractor.load_state_dict({k: v for k, v in state_dict.items() if k.startswith('lstm')})



def load_volleyball_dataset(dataset_root):
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)
    return videos_annot


dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"
videos_annot = load_volleyball_dataset(dataset_root)



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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_feature(video_ids, videos_annot, feature_root, feature_extractor):
    model.to(device)
    feature_extractor.eval()
    
    data = []
    for video_id in video_ids:
        # print(video_id)
        video_dir = os.path.join(feature_root, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        clips = os.listdir(video_dir)
        clips.sort()

        for clip in clips:
            clip_path = os.path.join(video_dir, clip)
            
            if clip_path.endswith('.npy'):
                features_dict = np.load(clip_path, allow_pickle=True).item()
                
                player_features = []
                for i in range(len(features_dict)):
                    with torch.no_grad():
                        # Convert to tensor and send to device
                        seq = torch.tensor(features_dict[i][1]).to(device)
                        seq_len,batch_size, feature_dim1, feature_dim2, num_players = seq.shape
                        seq = seq.view(batch_size, seq_len, feature_dim1)
                        extracted_features = feature_extractor(seq)
                        
                        feature_dim = extracted_features.shape[-1]
                        extracted_features = extracted_features.view(feature_dim)
                        player_features.append(extracted_features)
                player_features = torch.stack(player_features, dim=1) 
                max_pooled_features, _ = torch.max(player_features, dim=1)  
                
                category = videos_annot[str(video_id)][str(clip[:-4])]['category']
                category_idx = category_mapping.get(category)
                data.append((max_pooled_features.cpu().numpy(), category_idx))
    
    return data







# train_features = load_feature(train_videos,videos_annot,feature_root, feature_extractor)
# val_features = load_feature(val_videos,videos_annot,feature_root, feature_extractor)
# test_features = load_feature(test_videos,videos_annot,feature_root, feature_extractor)

# # Save the extracted features and labels
# torch.save(train_features, 'train_features_1024_lstm_on_pl.pth')
# torch.save(val_features, 'val_features_1024_lstm_on_pl.pth')
# torch.save(test_features, 'test_features_1024_lstm_on_pl.pth')




train_features=torch.load('train_features_1024_lstm_on_pl.pth')
val_features=torch.load('val_features_1024_lstm_on_pl.pth')
test_features=torch.load('test_features_1024_lstm_on_pl.pth')




train_loader = DataLoader(train_features, batch_size=32, shuffle=True,)
val_loader = DataLoader(val_features, batch_size=32, shuffle=False)
test_loader = DataLoader(test_features, batch_size=32, shuffle=False)





class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc4 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.sig(self.fc1(x))
        # x = self.sig(self.fc2(x))
        # x = self.sig(self.fc3(x))
        x = self.fc4(x)
        return x


input_dim = 1024  
num_classes = len(category_mapping)  
model = SimpleNN(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.AdamW(model.parameters(), lr=0.00001)



num_epochs = 1000
best_val_accuracy = 0.0

best_model_path='b4_nn_model'
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
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
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved with Val Accuracy: {val_accuracy:.2f}%")
        
model.load_state_dict(torch.load(best_model_path))
# Testing
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')



#     64