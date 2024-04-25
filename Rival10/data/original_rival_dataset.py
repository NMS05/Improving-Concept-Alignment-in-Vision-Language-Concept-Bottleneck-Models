import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm

"""
Original Dataset() class for the RIVAL-10 dataset
Refer to https://github.com/mmoayeri/RIVAL10/blob/gh-pages/datasets/local_rival10.py 
"""

_DATA_ROOT = 'data/RIVAL10/{}/'
_LABEL_MAPPINGS = 'data/RIVAL10/meta/label_mappings.json'
_WNID_TO_CLASS = 'data/RIVAL10/meta/wnid_to_class.json'

_ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
              'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy',
              'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']

def attr_to_idx(attr):
    return _ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return _ALL_ATTRS[idx]

def resize(img):
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img

class LocalRIVAL10(Dataset):
    def __init__(self, train=True, masks_dict=True):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.

        See __getitem__ for more documentation.
        '''
        self.train = train
        self.data_root = _DATA_ROOT.format('train' if self.train else 'test')
        self.masks_dict = masks_dict

        self.instance_types = ['ordinary']
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

        with open(_LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(_WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self,):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path+'/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))

            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)

            transformed_imgs.append(img)

        return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()
            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255*mask)))

        transformed_imgs = self.transform(imgs)
        img = transformed_imgs.pop(0)
        merged_mask = transformed_imgs.pop(0)
        out = dict({'img':img,
                    'attr_labels': attr_labels,
                    'changed_attrs': changed_attrs,
                    'merged_mask' :merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})
        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(_ALL_ATTRS)+1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)

        return out
    

"""
Explore dataset class
"""
# train_dataset = LocalRIVAL10(train=True, masks_dict=False)
# test_dataset = LocalRIVAL10(train=False, masks_dict=False)
# print(len(train_dataset), len(test_dataset))

# data_index = int(np.random.choice(np.arange(len(train_dataset)),size=1).item())
# img, attr_labels, cls_name, cls_label = train_dataset[data_index]['img'], train_dataset[data_index]['attr_labels'], train_dataset[data_index]['og_class_name'], train_dataset[data_index]['og_class_label']
# print("\n Class Name - {},  Class label - {} \n".format(cls_name, cls_label))
# for i,j in zip(_ALL_ATTRS,attr_labels): 
#     print(i,j.item())