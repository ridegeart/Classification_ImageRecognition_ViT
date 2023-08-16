'''Pytorch dataset loading script.
'''

import os
import pickle
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset
from level_dict import hierarchy,hierarchy_two
from helper import read_meta
import torch
from torchvision import transforms

class LoadDataset(Dataset):
    '''Reads the given csv file and loads the data.
    '''


    def __init__(self, csv_path, cifar_metafile, image_size=32, image_depth=3, return_label=True, transform=None):
        '''Init param.
        '''

        assert os.path.exists(csv_path), 'The given csv path must be valid!'

        self.csv_path = csv_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.meta_filename = cifar_metafile
        self.transform = transform
        self.data_list = self.csv_to_list()
        self.coarse_labels, self.fine_labels ,self.third_labels= read_meta(self.meta_filename)
        self.image_name_list = self.data_to_imagename()

        #check if the hierarchy dictionary is consistent with the csv file
        for k,v in hierarchy.items():
            if not k in self.coarse_labels:
                print(f"Superclass missing! {k}")
            for subclass in v:
                if not subclass in self.fine_labels:
                    print(f"Subclass missing! {subclass}")

        #check if the hierarchy_two dictionary is consistent with the csv file
        for k,v in hierarchy_two.items():
            if not k in self.fine_labels:
                print(f"Superclass missing! {k}")
            for subclass in v:
                if not subclass in self.third_labels:
                    print(f"Subclass missing! {subclass}")


    def csv_to_list(self):
        '''Reads the path of the file and its corresponding label 
        '''

        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def data_to_imagename(self):
        '''Reads csv_to_list , data for imagename'''
        imagelist=[]
        for i in self.data_list:
            imgsname = i[0]
            imagelist.append(imgsname)
        return imagelist

    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.data_list)

    def __getitem__(self, idx):
        '''Returns a single item.
        '''
        image_path, image, superclass, subclass ,subtwoclass = None, None, None, None, None #add subtwoclass
        if self.return_label:
            image_path, superclass, subclass ,subtwoclass= self.data_list[idx] #add subtwoclass
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            img = Image.open(image_path)
        else:
            img = Image.open(image_path)

        #if self.image_size != 32:
         #   cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            img = self.transform(img)

        if self.return_label:#add 
            return {
                'image':img,
                'label_1': self.coarse_labels.index(superclass.strip(' ')),
                'label_2': self.fine_labels.index(subclass.strip(' ')),
                'label_3': self.third_labels.index(subtwoclass.strip(' ')), #add subtwoclass
                'image_path':image_path
            }
        else:
            return {
                'image':img
            }
    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        label_1 = torch.tensor([item['label_1'] for item in batch])
        label_2 = torch.tensor([item['label_2'] for item in batch])
        label_3 = torch.tensor([item['label_3'] for item in batch])
        image_paths = [item['image_path'] for item in batch]

        # Convert images to a tensor and stack them
        images = torch.stack(images, dim=0)

        return {
            'image': images,
            'label_1': label_1,
            'label_2': label_2,
            'label_3': label_3,
            'image_path': image_paths
        }


