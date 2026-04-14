import os
import torch
import numpy as np
from PIL import Image as Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from random import random, choice, shuffle, sample, randint
from glob import glob
from scipy.ndimage.filters import gaussian_filter
import cv2

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def get_list(paths, must_contain='', return_num=False):
    
    image_list = []
    num_list = []
    for path in paths:
        tmps_list=[]
        extensions = ['jpg', 'bmp', 'png']
        for ext in extensions:
            tmp_list = glob(os.path.join(path, f'**/**.{ext}'), recursive=True)
            tmps_list.extend(tmp_list)
            # print(tmp_list)
        image_list.extend(tmps_list)    
        num_list.extend([len(tmps_list)])
    if return_num:    
        # print(num_list)
        return image_list, num_list
    return image_list


class FingerprintDataset(Dataset):

    def __init__(self, paths, opt):
        try:
            real_path = paths["real"]
            fake_path = paths["fake"]
        except:
            print("the data paths should contains real data path and fake data path, and should be composed to a dict{\"real\": real data path, \"fake\": fake data path}.")
        
        real_list = get_list(real_path)
        tmp_fake_list, num_fake = get_list(fake_path, return_num=True) 
        num_list = min(len(real_list), len(tmp_fake_list))
        real_list = list(zip(sample(real_list, num_list), [-1]*num_list))
        fake_list = []
        cur = 0
        for idx, num in enumerate(num_fake):
            # print(num)
            # print(type(num))
            # tmp_num = num_list
            tmp_num = num_list//(2*idx+1)
            fake_list.extend(list(zip(sample(tmp_fake_list[cur:cur+num], tmp_num), [idx]*tmp_num)))
            cur = cur+num
            
        print(f'Finding {len(real_list)} images in {real_path}')  
        print(f'Finding {len(fake_list)} images in {fake_path}')  
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        
            
        if opt.flip:
            flip_func = transforms.RandomHorizontalFlip()
            flip_v_func = transforms.RandomVerticalFlip()
        else: # no flip
            flip_func = transforms.Lambda(lambda img: img)
            flip_v_func = transforms.Lambda(lambda img: img)
        if opt.data_aug:
            aug_func = transforms.Lambda(lambda img: data_augment(img, opt))
        else: #no data augment
            aug_func = transforms.Lambda(lambda img: Image.fromarray(img))
        
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: custom_padding_reflect(img, opt)),
                aug_func,
                flip_func,
                flip_v_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        
        
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        # detail_label -1:real, 0:fake_data_name[0], 1:fake_data_name[1], 2:fake_data_name[2], 3:.....
        img_path, detail_label = self.total_list[idx]
        if detail_label != -1: # fake
            multi_label = torch.tensor([0,1,0])
            label = 1
        else: # real
            multi_label = torch.tensor([1,0,0])
            label = 0
        img = np.array(Image.open(img_path).convert('RGB'))
        img = self.transform(img)
        
        return img, multi_label, label, detail_label

class DiscriminatorDataset(Dataset):

    def __init__(self, paths, opt):
        try:
            real_path = paths["real"]
            fake_path = paths["fake"]
        except:
            print("the data paths should contains real data path and fake data path, and should be composed to a dict{\"real\": real data path, \"fake\": fake data path}.")
        
        real_list = get_list(real_path)
        tmp_fake_list, num_fake = get_list(fake_path, return_num=True) 
        num_list = min(len(real_list), len(tmp_fake_list))
        real_list = list(zip(sample(real_list, num_list), [-1]*num_list))
        fake_list = []
        cur = 0
        for idx, num in enumerate(num_fake):
            # print(num)
            # print(type(num))
            # tmp_num = num_list
            tmp_num = num_list//(4*idx+1)
            fake_list.extend(list(zip(sample(tmp_fake_list[cur:cur+num], tmp_num), [idx]*tmp_num)))
            cur = cur+num
            
        print(f'Finding {len(real_list)} images in {real_path}')  
        print(f'Finding {len(fake_list)} images in {fake_path}') 
        self.real_list = real_list
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        
            
        if opt.flip:
            flip_func = transforms.RandomHorizontalFlip()
            flip_v_func = transforms.RandomVerticalFlip()
        else: # no flip
            flip_func = transforms.Lambda(lambda img: img)
            flip_v_func = transforms.Lambda(lambda img: img)
        if opt.data_aug:
            aug_func = transforms.Lambda(lambda img: data_augment(img, opt))
        else: #no data augment
            aug_func = transforms.Lambda(lambda img: Image.fromarray(img))
        
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: custom_padding_reflect(img, opt)),
                aug_func,
                flip_func,
                flip_v_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        
        
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        # detail_label -1:real, 0:fake_data_name[0], 1:fake_data_name[1], 2:fake_data_name[2], 3:.....
        img_path, detail_label = self.total_list[idx]
        if detail_label != -1: # fake
            multi_label = torch.tensor([0,1,0])
            label = 1
        else: # real
            multi_label = torch.tensor([1,0,0])
            label = 0
        img = np.array(Image.open(img_path).convert('RGB'))
        img = self.transform(img)
        
        real_img_path, _ = choice(self.real_list)
        real_img = np.array(Image.open(real_img_path).convert('RGB'))
        real_img = self.transform(real_img)
        
        return img, multi_label, label, detail_label, real_img


class TripletDataset(Dataset):
    def __init__(self, paths, opt):
        try:
            real_path = paths["real"]
            fake_path = paths["fake"]
        except:
            print("the data paths should contains real data path and fake data path, and should be composed to a dict{\"real\": real data path, \"fake\": fake data path}.")
        
        real_list = get_list(real_path)
        tmp_fake_list, num_fake = get_list(fake_path, return_num=True) 
        num_list = min(len(real_list), len(tmp_fake_list))
        real_list = list(zip(sample(real_list, num_list), [-1]*num_list))
        fake_list = []
        cur = 0
        for idx, num in enumerate(num_fake):
            tmp_num = num_list
            # tmp_num = num_list//(4*idx+1)
            fake_list.extend(list(zip(sample(tmp_fake_list[cur:cur+num], tmp_num), [idx]*tmp_num)))
            cur = cur+num
            
        print(f'Finding {len(real_list)} images in {real_path}')  
        print(f'Finding {len(fake_list)} images in {fake_path}')  
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        
            
        if opt.flip:
            flip_func = transforms.RandomHorizontalFlip()
            flip_v_func = transforms.RandomVerticalFlip()
        else: # no flip
            flip_func = transforms.Lambda(lambda img: img)
            flip_v_func = transforms.Lambda(lambda img: img)
        if opt.data_aug:
            aug_func = transforms.Lambda(lambda img: data_augment(img, opt))
        else: #no data augment
            aug_func = transforms.Lambda(lambda img: Image.fromarray(img))
        
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.transform = transforms.Compose([
                transforms.Lambda(lambda img: custom_padding_reflect(img, opt)),
                aug_func,
                flip_func,
                flip_v_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        
        
    def __len__(self):
        return len(self.total_list)
    def __getitem__(self, idx):
        # detail_label -1:real, 0:fake_data_name[0], 1:fake_data_name[1], 2:fake_data_name[2], 3:.....
        img_path, detail_label = self.total_list[idx]
        if detail_label != -1: # fake
            multi_label = torch.tensor([0,1,0])
            label = 1
        else: # real
            multi_label = torch.tensor([1,0,0])
            label = 0
        img = np.array(Image.open(img_path).convert('RGB'))
        img = self.transform(img)
        
        return img, multi_label, label, detail_label
    def __getitem__(self, idx):
        """
            return anchor, positive, negative
        """
        img_path = str(self.total_list[idx])
        multi_label = self.labels_dict[img_path]
        if multi_label[0] == 1:
            label = 0
        else:
            label = 1
                  
        anchor = np.array(Image.open(img_path).convert('RGB'))
        anchor = self.transform(anchor)
        
        # another image 
        if label == 0: ## anchor is real
            img_path_1 = choice(self.real_list)
            img_path_2 = choice(self.fake_list)
        elif label == 1: ## anchor is fake
            img_path_1 = choice(self.fake_list)
            img_path_2 = choice(self.real_list)
        
        positive = np.array(Image.open(img_path_1).convert('RGB'))
        positive = self.transform(positive)
        
        negative = np.array(Image.open(img_path_2).convert('RGB'))
        negative = self.transform(negative)
        
        return anchor, multi_label, label, positive, negative
        



class PatchDataset(Dataset):

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
        
        # print(f'Finding {len(real_list)} images in {real_path}')  
        # print(f'Finding {len(fake_list)} images in {fake_path}')  
        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = torch.tensor([1,0,0])
        for i in fake_list:
            self.labels_dict[i] = torch.tensor([0,1,0])
                    
        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        
        self.pad_transform = transforms.Lambda(lambda img: custom_padding_reflect(img, opt))
        self.divide_patch = transforms.Lambda(lambda img: divide_patch(img, opt))
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        
        
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        # read image
        img_path = str(self.total_list[idx])
        multi_label = self.labels_dict[img_path]
        img = np.array(Image.open(img_path).convert('RGB'))
        # padding image    
        img = self.pad_transform(img)
        # divide patch
        patch1, patch2, patch3 = self.divide_patch(img)

        # transform
        img = self.transform(img)
        patch1 = self.transform(patch1)
        patch2 = self.transform(patch2)
        patch3 = self.transform(patch3)
        
        if multi_label[0] == 1:
            label = 0
        else:
            label = 1
        
        return img, patch1, patch2, patch3, label, multi_label


def divide_patch(img:np.array, opt):
    
    patch1 = img[:48, :, :]
    start = randint(50, 91)
    patch2 = img[start:start+48, :, :]
    patch3 = img[-48:, :, :]
        
    return patch1, patch2, patch3


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    # gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': cv2_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])

       
def custom_padding_white(img:np.array, opt):
    # Create a new blank image with the desired size and fill white color
    new_image = np.full((opt.loadSize[0], opt.loadSize[1], 3), 255, dtype=np.uint8)
    # Calculate the position to paste the input image
    left = (opt.loadSize[1] - img.shape[1]) // 2
    top = (opt.loadSize[0] - img.shape[0]) // 2
    # fill in
    new_image[top:top+img.shape[0], left:left+img.shape[1], :] = img
    
    return new_image

def custom_padding_reflect(img:np.array, opt):
    if img.shape[0] == 48 and img.shape[1] == 48:
        return img
    # Calculate padding sizes
    pad_height = (opt.loadSize[0] - img.shape[0]) // 2
    pad_width = (opt.loadSize[1] - img.shape[1]) // 2
    # Pad the image with reflect
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0,0)), mode='reflect')
    
    return padded_image
        
    
