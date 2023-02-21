'''
Youbao Tang
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
April 2019

For testing, you need to modify some arguments according to your own setting and run the command "python test.py".
'''



import argparse
import numpy as np
import torchvision
import torch
from torch.utils import data
from networks.xlsor import XLSor
from dataset.datasets import XRAYDataTestSet
import os
from PIL import Image as PILImage
from dataset.covid_dataset import test_clean_dataset, COVID19Dataset
from dataset.cgmh_dataset import CGMHTestSet
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
NUM_CLASSES = 1

DATA_DIRECTORY = '/home/project/Medical-Seg-Dataset-Distillation/cgmh_dataset/CGMH_PelvisSegment'
DARA_CSV_PATH = '/home/project/Medical-Seg-Dataset-Distillation/covid-chestxray-dataset/metadata.csv'
DATA_LIST_PATH = 'png'
RESTORE_FROM = './XLSor_snapshots/XLSor_cgmh_10000.pth'

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--csv-path", type=str, default=DARA_CSV_PATH)
    return parser.parse_args()


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    model = XLSor(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    dst = CGMHTestSet(args.data_dir)
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = [0 for i in range(len(dst))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - 0.9, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    dataset = torch.utils.data.Subset(dst, train_indices)
    testloader = data.DataLoader(dataset=dataset, 
                    batch_size=args.batch_size, shuffle=True, num_workers=4)

    interp = nn.Upsample(size=(256,256), mode='bilinear', align_corners=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    nums = 500
    turn = torchvision.transforms.ToPILImage()
    for num in range(0,nums,args.batch_size):
        for index, batch in enumerate(testloader):
            image = batch
            with torch.no_grad():
                prediction = model(image.cuda(), args.recurrence)
                if isinstance(prediction, list):
                    prediction = prediction[0]
                prediction = interp(prediction)
            for j in range(prediction.shape[0]):
                s_image = image[j] + 0.5
                print(s_image.max(),s_image.min())
                s_label = (prediction[j]>0.5).float()
                turn(s_image).save(os.path.join("./outputs",f"image_{num+j}.png"))
                turn(s_label).save(os.path.join("./outputs",f"mask_{num+j}.png"))


if __name__ == '__main__':
    main()
