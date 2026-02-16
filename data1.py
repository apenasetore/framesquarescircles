#----------------------------------------------------------
# Copyright (C) 2025 Maria Pazzini
# Embedded Systems & Instrumentation Department
# École Supérieure d'Ingénieurs - ESIGELEC, France
# This software is intended for research purposes only;
# its redistribution is forbidden under any circumstances.
#----------------------------------------------------------
import os
import sys
import csv
import cv2
import glob
import h5py
import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

#==========================================================

class SignalDataset(Dataset):
    def __init__(self, data_path, data_type, noise=None, nfold=0, load_first=False):
        self.noise = noise
        self.nfold = nfold
        self.load_first = load_first
        
        self.data_list = glob.glob(os.path.join(data_path,'*.mat'))
        self.label_list = glob.glob(os.path.join(data_path,'*_label.*')) 
        self.image_list = glob.glob(os.path.join(data_path,'*_image.*')) 

        self.data_list.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        self.data_list = self.filter_datalist(self.data_list, data_type)

        self.label_list.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.label_list = self.filter_datalist(self.label_list, data_type)

        self.image_list.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.image_list = self.filter_datalist(self.image_list, data_type)

        if self.load_first:
            random_numbers = [random.uniform(0.0, 1.0) for _ in range(len(self.data_list))]
            self.data_list = [self.load_signal(n, random_numbers[idx]) for idx, n in enumerate(self.data_list)]
            self.label_list = [self.load_image(n, random_numbers[idx]) for idx, n in enumerate(self.label_list)]
            self.image_list = [self.load_image(n, random_numbers[idx], 'image') for idx, n in enumerate(self.image_list)]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label_file = self.label_list[idx]
        image_file = self.image_list[idx]

        rand = random.random()

        if not self.load_first:
            signal = self.load_signal(self.data_list[idx], rand)
            label = self.load_image(label_file, rand)
            image = self.load_image(image_file, rand, 'image')

            return signal, label, image
        else:
            data = self.data_list[idx]
            return data, label_file, image_file
        

    def filter_datalist(self, data_list, filter_type):
        indexes = self.get_index()
        filtered_files = []

        if filter_type == 'train':
            del indexes[self.nfold]

        for file_path in data_list:
            file_name = os.path.basename(file_path).split('_')[1].split('.')[0]
            
            try:
                file_prefix = int(file_name)
            except ValueError:
                continue
            if filter_type == 'test':
                if file_prefix in indexes[self.nfold]:
                    filtered_files.append(file_path)

            if filter_type == 'train':
                for index_array in indexes:
                    if file_prefix in index_array:
                        filtered_files.append(file_path)
        
        return filtered_files

    def get_index(self, path_indices='indices.mat'):
        with h5py.File(path_indices, 'r') as f:
            index_array = f['parts']

            index_data = [f[index_ref[0]][:] for index_ref in index_array]
            indexes = [np.transpose(index).astype(int).squeeze() for index in index_data]

            # for idx, index in enumerate(indexes):
            #     print(f"Transposed index {idx}:")
            #     print(index)

            return indexes

    def load_signal(self, filename, rand):
        data = h5py.File(filename, 'r')
        signal = data.get('IQm')
        signal = np.array(signal).transpose(1,2,0)
        
        padd_left = math.floor((2 ** math.ceil(math.log2(signal.shape[2])) - signal.shape[2])/2)
        padd_right = math.ceil((2 ** math.ceil(math.log2(signal.shape[2])) - signal.shape[2])/2)
        padd_bottom = 2 ** math.ceil(math.log2(signal.shape[1])) - signal.shape[1]

        padded_signal = np.pad(
            signal,
            pad_width=((0, 0), (0, padd_bottom), (padd_left, padd_right)),
            mode='constant',  
            constant_values=0 
        )

        if self.noise is not None:
            sigma = np.random.uniform(high=self.noise)
            noise = np.random.normal(scale=sigma, size=padded_signal.shape)
            noise_signal = padded_signal + noise
            
            min_val = noise_signal.min()
            max_val = noise_signal.max()
            normalized_signal = (noise_signal - min_val) / (max_val - min_val)

            if rand > 0.5:
               normalized_signal = np.flip(normalized_signal, axis=2)  
               normalized_signal = np.flip(normalized_signal, axis=0)
            return normalized_signal.astype(np.float32)

        return padded_signal.astype(np.float32)

    def load_image(self, filename, rand, image_type='label'):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.noise is not None and rand > 0.5:
            image = np.flip(image, axis=1)

        image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)/255 

        if self.noise is not None and image_type == 'image':
            sigma = np.random.uniform(high=self.noise)
            noise = np.random.normal(scale=sigma, size=image.shape)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        return image

#----------------------------------------------------------

'''
data_path = '../signal_segmentation/dataset'

data2 = SignalDataset(data_path, data_type='train', noise=0.025, nfold=1, load_first=False)
data = SignalDataset(data_path, data_type='test', nfold=1, load_first=False)
print(len(data2.label_list))
print(len(data2.data_list))
print(data2.__getitem__(0)[0].shape)
print(data2.__getitem__(0)[1].shape)
print(data2.__getitem__(0)[2].shape)


print(len(data.label_list))
print(len(data.data_list))


#cv2.imshow('Image', data.__getitem__(0)[0])
#cv2.imshow('Label', label.data_list[0])

'''