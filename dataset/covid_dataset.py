import torchxrayvision as xrv
from .stnaugment import STNAugment
import os, sys, math, random, torch
import zipfile
import imageio
from PIL import Image
from torchvision import transforms
import numpy as np
from torchxrayvision.datasets import apply_transforms
from torch.utils.data import DataLoader, Dataset
import os.path as osp
import glob
import numpy as np
import random
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt

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

def normalize(img, reshape=False, z_norm=False):
    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # add color channel
        img = img[None, :, :]
    img = torch.from_numpy(img.astype(np.float32) / 255)
    if z_norm:
        img = 2 * img - 1.
    return img


class COVID19Dataset(xrv.datasets.COVID19_Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 views=["PA", "AP"],
                 transform=None,
                 semantic_masks=False,
                 scale = True,
                 ):
        super(COVID19Dataset, self).__init__(
            imgpath=imgpath,
            csvpath=csvpath,
            views=views,
            transform=transform,
            semantic_masks=semantic_masks
        )
        self.scale = scale
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    def get_semantic_mask_dict(self, image_name):

        archive_path = "semantic_masks_v7labs_lungs/" + image_name
        semantic_masks = {}
        if archive_path in self.semantic_masks_v7labs_lungs_namelist:
            with zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path).open(archive_path) as file:
                mask = imageio.imread(file.read())
                mask = Image.fromarray(mask).convert("L")
                semantic_masks["Lungs"] = mask

        return semantic_masks
    
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = Image.open(img_path).convert('L')
        sample["img"] = img
        if self.semantic_masks:
            sample["semantic_masks"] = self.get_semantic_mask_dict(imgid)
        sample = apply_transforms(sample, self.transform)
        mask = (sample["semantic_masks"]["Lungs"] == 1.).float()
        sample["semantic_masks"]["Lungs"] = mask
        return sample


class CleanCOVID19Dataset(Dataset):
    def __init__(self, samples, dataset):
        self.samples = samples
        self.dataset = dataset
        self.scale = True
        self.crop_h, self.crop_w = 256,256
        self.mean = 0.5
        self.is_mirror = True
    def __len__(self):
        return len(self.samples)
    
    def generate_scale_label(self, image, label):
        f_scale = 0.35 + random.random() * 0.9
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def tensor_to_cv2(self,image):
        ndarray = image.cpu().numpy().transpose((1,2,0))
        image = cv2.convertScaleAbs(ndarray, alpha=(255.0))
        return image
    
    def __getitem__(self, item):
        idx = self.samples[item]
        sample = self.dataset[idx]
        # img: torch.Tensor mask: torch.Tensor
        sample["img"] = self.tensor_to_cv2(sample["img"])
        sample["semantic_masks"]["Lungs"] = self.tensor_to_cv2(sample["semantic_masks"]["Lungs"])
        angle = -15.0 + random.random() * 30.0
        sample["img"] = rotate_bound(sample["img"], angle, cv2.INTER_CUBIC)
        sample["semantic_masks"]["Lungs"] = rotate_bound(sample["semantic_masks"]["Lungs"], angle, cv2.INTER_CUBIC)/255
        image,label = sample["img"], sample["semantic_masks"]["Lungs"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        top_p = random.randint(0, pad_h)
        left_p = random.randint(0, pad_w)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, top_p, pad_h - top_p, left_p,
                                         pad_w - left_p, cv2.BORDER_CONSTANT,
                                         value=(0.,))
            label_pad = cv2.copyMakeBorder(label, top_p, pad_h - top_p, left_p,
                                           pad_w - left_p, cv2.BORDER_CONSTANT,
                                           value=(0.,))
        else:
            img_pad, label_pad = image, label
        img_pad /= 255.
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
        # print(label.max(),label.min(),label.mean(),image.max(),image.min(),image.mean())
        # import torchvision.transforms as T
        # turn = T.ToPILImage()
        # turn(label).save("mask.png")
        # turn(image).save("image.png")
        return image, label
    
def clean_dataset(dataset):
    assert dataset.semantic_masks, "only turn segmentation task"
    samples = []
    for idx in range(len(dataset)):
        imgid = dataset.csv['filename'].iloc[idx]
        archive_path = "semantic_masks_v7labs_lungs/" + imgid
        if archive_path in dataset.semantic_masks_v7labs_lungs_namelist:
            samples.append(idx)
    return CleanCOVID19Dataset(samples, dataset)

def test_clean_dataset(dataset):
    assert dataset.semantic_masks, "only turn segmentation task"
    samples = []
    for idx in range(len(dataset)):
        imgid = dataset.csv['filename'].iloc[idx]
        archive_path = "semantic_masks_v7labs_lungs/" + imgid
        if archive_path in dataset.semantic_masks_v7labs_lungs_namelist:
            samples.append(idx)
    return CleanCOVID19TestSet(samples, dataset)   
class CleanCOVID19TestSet(data.Dataset):
    def __init__(self, samples, dataset):
        self.samples = samples
        self.dataset = dataset
        self.scale = True
        self.crop_h, self.crop_w = 256,256
        self.mean = 0.5
        self.is_mirror = True
    def __len__(self):
        return len(self.samples)
    
    def generate_scale_label(self, image, label):
        f_scale = 0.35 + random.random() * 0.9
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def tensor_to_cv2(self,image):
        ndarray = image.cpu().numpy().transpose((1,2,0))
        image = cv2.convertScaleAbs(ndarray, alpha=(255.0))
        return image

    def __getitem__(self, index):
        idx = self.samples[index]
        sample = self.dataset[idx]
        # img: torch.Tensor mask: torch.Tensor
        sample["img"] = self.tensor_to_cv2(sample["img"])
        image = sample["img"]
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