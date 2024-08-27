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


annot_root= r"Z:\work\cnn\volley_project\data_set\volleyball_tracking_annotation\volleyball_tracking_annotation\_"

video_root=r"Z:\work\cnn\volley_project\data_set\volleyball_\videos"


dataset_root = r"Z:\work\cnn\volley_project\data_set\volleyball_"


feature_root = r'Z:\work\cnn\volley_project\data_set\volleyball_\features_p\player-level\resnet'
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
all_videos=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
        ,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]

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







def load_feature(video_ids, feature_root):
    for video_id in video_ids:
        video_dir = os.path.join(feature_root, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        clips = os.listdir(video_dir)
        clips.sort()

        for clip in clips:
            clip_path = os.path.join(video_dir, clip)
            
            if clip_path.endswith('.npy'):
                features_dict = np.load(clip_path, allow_pickle=True).item()
                
                for i in range(9):
                    player_features = []
                    for z in range(len(features_dict)):
                        seq = torch.tensor(features_dict[z][1][i])
                        _, dnn, _, _ = seq.shape
                        seq = seq.view(dnn, -1)
                        player_features.append(seq)
                    
                    player_features = torch.stack(player_features, dim=1) 
                    max_pooled_features, _ = torch.max(player_features, dim=1)
                    
                    save_dir = f"Z:/work/cnn/volley_project/data_set/volleyball_/for_b5/{video_id}/{clip}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_path = os.path.join(save_dir, f"frame_{i}.pt")
                    torch.save(max_pooled_features, save_path)


# load_feature(all_videos,feature_root)



def load_data(video_ids,video_root,videos_annot):
    data=[]
    for video_id in video_ids:
        video_dir = os.path.join(video_root, str(video_id))
        if not os.path.isdir(video_dir):
            continue

        clips = os.listdir(video_dir)
        clips.sort()

        for clip in clips:
            clip_path = os.path.join(video_dir, clip)
            if not os.path.isdir(clip_path):
                continue
            
            
            aggregated_features=[]
            for i in range(9):
                path=fr"Z:\work\cnn\volley_project\data_set\volleyball_\for_b5\{video_id}\{clip}.npy\frame_{i}.pt"
                features = torch.load(path)
                aggregated_features.append(features.unsqueeze(0))
            aggregated_features = torch.cat(aggregated_features, dim=0)
            category = videos_annot[str(video_id)][str(clip)]['category']
            category_idx = category_mapping.get(category)
            
            data.append((aggregated_features, category_idx))
    return data






train_features = load_data(train_videos,video_root, videos_annot)
val_features = load_data(val_videos,video_root, videos_annot)
test_features = load_data(test_videos,video_root, videos_annot)

train_loader = DataLoader(train_features, batch_size=32, shuffle=True)
val_loader = DataLoader(val_features, batch_size=32, shuffle=False)
test_loader = DataLoader(test_features, batch_size=32, shuffle=False)


class VolleyballLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(VolleyballLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_dim = 2048  
hidden_dim = 1024   
num_layers = 1   
num_classes = len(category_mapping)  

model = VolleyballLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)



#72