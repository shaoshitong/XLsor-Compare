import torch.nn as nn
import numpy as np
import cv2,torch,os,sys,PIL,random
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

def rotate_bound(image, angle, flag):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flag)

class CGMHDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "Image/")
        self.label_path = os.path.join(self.root_path, "Label/")
        self.path_set = []
        for path in os.listdir(self.image_path):
            if path.endswith(".png"):
                self.path_set.append(os.path.join(self.image_path,path))
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((384, 384))])
        self.scale = True
        self.crop_h, self.crop_w = 256,256
        self.mean = 0.5
        self.is_mirror = True
    def __len__(self):
        return len(self.path_set)

    def generate_scale_label(self, image, label):
        f_scale = 0.67 + random.random() * 0.9
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def tensor_to_cv2(self,image):
        ndarray = image.cpu().numpy().transpose((1,2,0))
        image = cv2.convertScaleAbs(ndarray, alpha=(255.0))
        return image
    
    def __getitem__(self, item):
        path = self.path_set[item]
        image_path = path
        label_path = path.replace("Image/", "Label/")
        image = PIL.Image.open(image_path).convert("L")
        label = PIL.Image.open(label_path).convert("L")
        image = self.transform(image).float()
        label = (self.transform(label) > 0.5).float()
        image,label = self.tensor_to_cv2(image),self.tensor_to_cv2(label)
        angle = -15.0 + random.random() * 30.0
        image = rotate_bound(image, angle, cv2.INTER_CUBIC)
        label = rotate_bound(label, angle, cv2.INTER_CUBIC)
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        img_pad = image.astype(float)
        label_pad = label.astype(float)
        img_pad /= 255.
        label_pad /= 255.
        img_pad -= self.mean
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip]
            label = label[:, ::flip]
        label = torch.from_numpy(np.expand_dims(label, axis=0).copy())
        image = torch.from_numpy(np.expand_dims(image, axis=0).copy())
        return image, label

class CGMHTestSet(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "Image/")
        self.path_set = []
        for path in os.listdir(self.image_path):
            if path.endswith(".png"):
                self.path_set.append(os.path.join(self.image_path,path))
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((384, 384))])
        self.scale = True
        self.crop_h, self.crop_w = 256,256
        self.mean = 0.5
        self.is_mirror = True
    def __len__(self):
        return len(self.path_set)

    def generate_scale_label(self, image, label):
        f_scale = 0.67 + random.random() * 0.9
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def tensor_to_cv2(self,image):
        ndarray = image.cpu().numpy().transpose((1,2,0))
        image = cv2.convertScaleAbs(ndarray, alpha=(255.0))
        return image
    
    def __getitem__(self, item):
        path = self.path_set[item]
        image_path = path
        image = PIL.Image.open(image_path).convert("L")
        image = self.transform(image).float()
        image = self.tensor_to_cv2(image)
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        size = image.shape
        image = np.asarray(image, np.float32)
        image /= 255.
        image -= self.mean
        img_h, img_w = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=(0.0))
        image = torch.from_numpy(np.expand_dims(image, axis=0).copy())
        return image