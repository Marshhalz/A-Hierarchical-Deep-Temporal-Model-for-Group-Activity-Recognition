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




def get_split_indices(videos_annot, train_videos, val_videos, test_videos):
    split_dict = {}
    
    for video in videos_annot:
        for clip in videos_annot[video]:
            for frame_id, _ in videos_annot[video][clip]['frame_boxes_dct'].items():
                if int(video) in train_videos:
                    split_dict[int(frame_id)] = 'train'
                elif int(video) in val_videos:
                    split_dict[int(frame_id)] = 'val'
                elif int(video) in test_videos:
                    split_dict[int(frame_id)] = 'test'
                    
    return split_dict








class VolleyballDataset(Dataset):
    def __init__(self,dataset_root, videos_annot,split_dict, transform=None):
        self.videos_annot = videos_annot
        self.transform = transform
        self.data = []
        self.cat_dic={}
        majority_class = 8
        max_majority_samples = 10000
        self.cat_dic[majority_class] = 1
        self.split_dict = split_dict
        for video in videos_annot:
            for clip in videos_annot[video]:
                for frame_id, boxes_info in videos_annot[video][clip]['frame_boxes_dct'].items():
                    frame_path = f"{dataset_root}/videos/{video}/{clip}/{frame_id}.jpg"
                    if(int(clip)==int(frame_id)):
                        if os.path.exists(frame_path):
                            image = Image.open(frame_path).convert('RGB')
                            for box_info in boxes_info:
                                if int(box_info.frame_ID) == frame_id:
                                    x1, y1, x2, y2 = box_info.box
                                    player_category = box_info.category
                                    player_category = category_mapping.get(str(player_category))
                                    if player_category not in self.cat_dic and player_category != majority_class:
                                        self.cat_dic[player_category] = 1
                                    elif player_category != majority_class:
                                        self.cat_dic[player_category] += 1
                                    if player_category == majority_class:
                                        if self.cat_dic[majority_class] > max_majority_samples:
                                            if random.random() > (max_majority_samples / self.cat_dic[majority_class]):
                                                continue  
                                        else:
                                            self.cat_dic[majority_class] += 1
                                            cropped_image = image.crop((x1, y1, x2, y2))
                                            if self.transform:
                                                cropped_image = self.transform(cropped_image)
                                            player_category = torch.nn.functional.one_hot(torch.tensor(player_category), num_classes=len(category_mapping))
                                            split = self.split_dict.get(int(frame_id), None)
                                            self.data.append((cropped_image, player_category, split, frame_id))
                                    else:
                                        cropped_image = image.crop((x1, y1, x2, y2))
                                        if self.transform:
                                            cropped_image = self.transform(cropped_image)
                                        player_category = torch.nn.functional.one_hot(torch.tensor(player_category), num_classes=len(category_mapping))
                                        split = self.split_dict.get(int(frame_id), None)
                                        self.data.append((cropped_image, player_category, split, frame_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cropped_image, cat, split,frame_id_x= self.data[idx]
        

        
        return cropped_image, cat, split



train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]



def load_volleyball_dataset(dataset_root):
    with open(f'{dataset_root}/annot_all.pkl', 'rb') as file:
        videos_annot = pickle.load(file)
    return videos_annot


videos_annot = load_volleyball_dataset(dataset_root)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


split_dict = get_split_indices(videos_annot, train_videos, val_videos, test_videos)




dataset = VolleyballDataset(dataset_root, videos_annot, split_dict, transform=transform)



train_indices = []
val_indices = []
test_indices = []

for idx in range  (len(dataset)):
    if dataset.data[idx][2] == 'train':
        train_indices.append(idx)
    elif dataset.data[idx][2] == 'val':
        val_indices.append(idx)
    elif dataset.data[idx][2] == 'test':
        test_indices.append(idx)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = models.resnet152(pretrained=True)


num_classes = len(category_mapping)


model.fc = nn.Sequential(
    nn.Identity(),       
    nn.Linear(2048, 2024),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(2024, 2048),
    nn.ReLU(),
    nn.Linear(2048, num_classes)
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00000000001)

best_val_accuracy = 0.0
best_model_weights = None

num_epochs = 20
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
    torch.cuda.empty_cache()

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


    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()

# Save the best model weights
torch.save(best_model_weights, 'vanila_rsnt_model_on_players.pth')

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








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    with torch.no_grad():
        for frame_id, boxes_info in frame_boxes.items():
            try:
                img_path = os.path.join(clip_dir_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')
                
                preprocessed_images = []
                for box_info in boxes_info:
                    x1, y1, x2, y2 = box_info.box
                    cropped_image = image.crop((x1, y1, x2, y2))
                    
                    preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))
                preprocessed_images = torch.cat(preprocessed_images)
                preprocessed_images = preprocessed_images.to(device)
                dnn_repr = model(preprocessed_images)
                dnn_repr = dnn_repr.view(len(preprocessed_images), -1)  # 12 x 2048 

                dnn_repr_cpu = dnn_repr.cpu().numpy()
                
                np.save(output_file, dnn_repr_cpu)
            except Exception as e:
                print(f"An error occurred: {e}")







if __name__ == '__main__':


    model, preprocess = prepare_model()

    output_root = f'{dataset_root}/features/image-level/resnet'

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

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







output_root = r'Z:\work\cnn\volley_project\data_set\volleyball_\features_maxed\image-level\resnet'
feature_dirs = os.listdir(r"Z:\work\cnn\volley_project\data_set\volleyball_\features\image-level\resnet")
feature_dirs.sort()


os.makedirs(output_root, exist_ok=True)

for video_dir in feature_dirs:
    f_dir_path = os.path.join(r"Z:\work\cnn\volley_project\data_set\volleyball_\features\image-level\resnet", video_dir)
    if not os.path.isdir(f_dir_path):
        continue
    
    for frame_file in os.listdir(f_dir_path):
        if frame_file.endswith('.npy'):
            frame_path = os.path.join(f_dir_path, frame_file)
            loaded_features = np.load(frame_path)
            max_pooled_features = np.max(loaded_features, axis=0)
            
            output_file = os.path.join(output_root, video_dir)
            os.makedirs(output_file, exist_ok=True)
            output_file_path = os.path.join(output_file, frame_file)  
            np.save(output_file_path, max_pooled_features)



