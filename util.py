from genericpath import isdir
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import pandas as pd
import os
import cv2
from skimage import io
import numpy as np
import pickle
import yaml
from cerberus import Validator
import logging
import sys
from PIL import Image
import math
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
import joblib

def read_image(img_path):
    img = Image.open(img_path)

    return img


class ImageDataset(Dataset):
    def __init__(self, img_dir, features_dir, feature_scaler_path, class_names, transform_main=lambda x: x, transform_augment=lambda x: x):
        self.img_dir = img_dir
        self.transform_main = transform_main
        self.transform_augment = transform_augment
        self.class_names = class_names

        self.feature_scaler = joblib.load(feature_scaler_path)
        
        meta_path = f"metadata/{img_dir.replace('../', '')}_{feature_scaler_path.split('/')[-1].split('.')[0]}"

        logging.info(f"No metadata present in'{meta_path}'.")
        logging.info("Calculating metadata...")
            
        self.calculate_metadata(img_dir, features_dir)

    def calculate_metadata(self, img_dir, features_dir):
        self.img_name_to_label = dict()
        
        self.img_names = list()
        self.label_to_classname = dict()
        self.features = dict()

        for class_idx, class_name in enumerate(self.class_names):
            self.label_to_classname[class_idx] = class_name

            for f in sorted(os.listdir(f"{img_dir}/{class_name}")):
                name = f"{class_name}/{f}"
                label = class_idx

                self.img_names.append(name)

                self.img_name_to_label[name] = label

                try:
                    feature_name = f.split(".")[0] + "_FEATURES.txt"
                    self.features[name] = self.feature_scaler.transform(pd.read_csv(f"{features_dir}/{feature_name}", header=None, delimiter=",").values)[0]
                except:
                    feature_name = f.split(".")[0] + "P_FEATURES.txt"
                    self.features[name] = self.feature_scaler.transform(pd.read_csv(f"{features_dir}/{feature_name}", header=None, delimiter=",").values)[0]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = f"{self.img_dir}/{img_name}"

        image = read_image(img_path)
        image = self.transform_main(image)

        image = self.transform_augment(image)
        features = self.features[img_name]
        label = self.img_name_to_label[img_name]
        
        return image, features, label
