import os
import csv
import torch
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
import torch.nn as nn
from PIL import Image as Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from random import random, choice, shuffle, sample
from glob import glob
from scipy.ndimage.filters import gaussian_filter
import cv2
# from networks.center_classifier import ClusterClassifier

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def get_list(path, must_contain=''):
    
    image_list = []
    extensions = ['jpg', 'bmp', 'png']
    for ext in extensions:
        image_list.extend(glob(os.path.join(path, f'**/**.{ext}'), recursive=True))
      
    return image_list


class FingerprintDataset(Dataset):

    def __init__(self, paths, opt):
        try:
            real_path = paths["real"]
            fake_path = paths["fake"]
        except:
            print("the data paths should contains real data path and fake data path, and should be composed to a dict{\"real\": real data path, \"fake\": fake data path}.")
        
        real_list = get_list(real_path)
        fake_list = get_list(fake_path) 
        num_list = min(len(real_list), len(fake_list))
        real_list = sample(real_list, num_list)
        fake_list = sample(fake_list, num_list)
        
        print(f'Finding {len(real_list)} images in {real_path}')  
        print(f'Finding {len(fake_list)} images in {fake_path}')  
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = torch.tensor([1,0,0])
        for i in fake_list:
            self.labels_dict[i] = torch.tensor([0,1,0])
            

        aug_func = transforms.Lambda(lambda img: Image.fromarray(img))
        
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: custom_padding_reflect(img, opt)),
                aug_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        
        
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = str(self.total_list[idx])
        multi_label = self.labels_dict[img_path]
        img = np.array(Image.open(img_path).convert('RGB'))
        img = self.transform(img)
        if multi_label[0] == 1:
            label = 0
        else:
            label = 1
        return img, label


def custom_padding_reflect(img:np.array, opt):
    # Calculate padding sizes
    pad_height = (opt.loadSize[0] - img.shape[0]) // 2
    pad_width = (opt.loadSize[1] - img.shape[1]) // 2
    # Pad the image with reflect
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0,0)), mode='reflect')
    
    return padded_image
       
def custom_padding_white(img:np.array, opt):
    # Create a new blank image with the desired size and fill white color
    new_image = np.full((opt.loadSize[0], opt.loadSize[1], 3), 255, dtype=np.uint8)
    # Calculate the position to paste the input image
    left = (opt.loadSize[1] - img.shape[1]) // 2
    top = (opt.loadSize[0] - img.shape[0]) // 2
    # fill in
    new_image[top:top+img.shape[0], left:left+img.shape[1], :] = img
    
    return new_image
    

def create_dataloader(paths, opt):
    
    shuffle = True
    data_loader = DataLoader(
        FingerprintDataset(paths, opt),
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )
    
    return data_loader

# class ResNet50(nn.Module):
    # def __init__(self, opt, num_classes):
        # super(ResNet50, self).__init__()
        # model = resnet50(num_classes=num_classes) 
        # state_dict = torch.load('./checkpoints/example/model_epoch_best.pth', 
                                # map_location='cpu')
        # model.load_state_dict(state_dict['model'])
        # self.model = model
        # del(self.model.fc)
        # self.classifier = ClusterClassifier()
        
        
    # def forward(self, x):
        # out = self.model(x)
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # features = self.model.avgpool(x)
        # features = features.flatten()
        # out = self.classifier(features)
        # return out

class ResNet50(nn.Module):
    def __init__(self, opt, num_classes):
        super(ResNet50, self).__init__()
        model = resnet50(num_classes=num_classes) 
        state_dict = torch.load(opt.model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        self.model = model
        # print(self.model)
        
    def forward(self, x):
        out = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        features = self.model.avgpool(x)
        features = features.view(features.size(0),-1)
        # out = self.model.fc(features)
        
        return features
        
def validate(model, opt, return_feature=False):
       
    paths = {"real":os.path.join(opt.data_root, 'test', opt.real_data_name), 
             "fake":os.path.join(opt.data_root, 'test', opt.fake_data_name)}
             
    data_loader = create_dataloader(paths, opt)
    with torch.no_grad():
        y_true, y_pred, features_list = [], [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            features = model(in_tens)
            features = features.tolist()
            features_list.extend(features)
            y_true.extend(label.flatten().tolist())
            

    y_true = np.array(y_true)
    features_list = np.array(features_list)
    return y_true, features_list




# Running tests
opt = TestOptions().parse(print_options=True)
model_name = opt.name
# rows = [["{} model testing on...".format(model_name)],
        # ['testset', 'accuracy', 'avg precision', 'real acc', 'fake acc']]

os.makedirs(f'./features_visualize/ablation/{model_name}', exist_ok=True)

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    model = ResNet50(opt, num_classes=1)
    # model = resnet50(num_classes=1)
    # state_dict = torch.load(opt.model_path, map_location='cpu')
    # model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()
    # print(model)
    opt.fake_data_name = val
    y_true, features_list = validate(model, opt)
    # rows.append([val, acc, ap, r_acc, f_acc])
    # print("({}) r_acc: {}; f_acc: {}; acc: {}; ap: {}".format(val, r_acc, f_acc, acc, ap))
    
    if not os.path.isfile(f'./features_visualize/ablation/{model_name}/real.npy'):
        np.save(f'./features_visualize/ablation/{model_name}/real.npy', features_list[y_true==0])
    np.save(f'./features_visualize/ablation/{model_name}/{val}.npy', features_list[y_true==1])

# csv_name = results_dir + '/{}.csv'.format(model_name)
# with open(csv_name, 'w') as f:
    # csv_writer = csv.writer(f, delimiter=',')
    # csv_writer.writerows(rows)
