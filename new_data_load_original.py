import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from torchvision.transforms import Normalize
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import random


class Dataset_dict(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=None,root_dir1=None, transform=None, start_freq=0, end_freq=29, which_freq=0, failed_file=".", random_choose_slice=True, channel3=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            which_freq (int): 主要被用来挑图
        """
        self.freq_start = start_freq
        self.freq_end = end_freq
        self.dict_root = root_dir
        self.dict_root1 = root_dir1
        self.transform = transform
        self.dict_data = []
        self.which_freq = which_freq
        self.failed_file = failed_file
        self.normalize = Normalize(mean=[0.5], std=[0.5])
        self.random_choose_slice = random_choose_slice
        self.channel3 = channel3
        #self.freq_1_or_5_channel = freq_1_or_5_channel
        # self.flag = "Val"
        # print("len of root dir:",len(os.walk(root_dir)))
        print("root dir:", root_dir)
        print("another root dir",root_dir1)

        for (root, dirs, files) in os.walk(root_dir):
            print("root:", root, "dirs:", dirs, "len of files:", len(files))

            self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
            self.root = root
        if root_dir1 != None:
            for (root, dirs, files) in os.walk(root_dir1):
                print("root:", root, "dirs:", dirs, "len of files:", len(files))

                self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
                self.root = root


    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self, idx):
        images_list = []
        clean_list = []
        ddr_list = []
        t60_list = []
        MeanT60_list = []
        name_list = []
        data_dict = dict()
        dict_data_name = self.dict_data[idx]
        audio_image_dict = dict()
        try:
            # print(dict_data_root)
            audio_image_dict = torch.load(dict_data_name)
        except:
            print("failed read file name:", dict_data_name)
            audio_image_dict = torch.load(self.failed_file)

        dict_keys = list(audio_image_dict.keys())[0]
        dict_data = audio_image_dict[dict_keys]
        valid_info_count = 0

        for list_c in range(len(dict_data)):
            if dict_data[list_c] == 0:
                continue
            else:
                valid_info_count += 1
                # if self.freq_1_or_5_channel==5:
                if "image" in dict_data[list_c].keys():
                    if isinstance(dict_data[list_c]['image'],list):#如果是list，说明是新数据，因为有5个频段
                        #temp_image = dict_data[list_c]['image']  # 选中对应的倍频程
                        ######################################################
                        temp_image = dict_data[list_c]['image'][self.which_freq]  # 选中对应的倍频程
                        ######################################################
                        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)


                        temp_clean = dict_data[list_c]['clean'][self.which_freq]  # 干净语谱图
                        #temp_clean = torch.from_numpy(temp_clean)
                        temp_clean = temp_clean[temp_clean.shape[0] // 2:].unsqueeze(0)

                        images_list.append(temp_image)  # 选图的上半部分，shape=[1, 65, 175] for 2khz, 175是因为只有一个slice(4s)
                        clean_list.append(temp_clean)

                        t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end], 0)) #by hsl
                        ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end], 0)) #by hsl
                        # t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][13], 0))#对应500hz
                        # ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][13], 0))
                        MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'], 0))
                        name_list.append(dict_data_name)
                    else:#老数据
                        temp_image = dict_data[list_c]['image']  # 选中对应的倍频程
                        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)

                        temp_clean = dict_data[list_c]['clean']
                        temp_clean = torch.from_numpy(temp_clean)
                        temp_clean = temp_clean[temp_clean.shape[0] // 2:].unsqueeze(0)

                        images_list.append(temp_image)  # 选图的上半部分，shape=[1, 65, 175] for 2khz, 175是因为只有一个slice(4s)
                        clean_list.append(temp_clean)

                        t60_list.append(
                            torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end], 0))  # by hsl
                        ddr_list.append(
                            torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end], 0))  # by hsl
                        # t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][13], 0))#对应500hz
                        # ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][13], 0))
                        MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'], 0))
                        name_list.append(dict_data_name)

                # elif self.freq_1_or_5_channel == 1:
                #     if "image" in dict_data[list_c].keys():



        images = torch.cat(images_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]
        cleans = torch.cat(clean_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]

        ddr = torch.cat(ddr_list, dim=0)
        t60 = torch.cat(t60_list, dim=0)
        MeanT60 = torch.cat(MeanT60_list, dim=0)

        if self.transform:
            images = self.transform(images)  # images:[3, 1, 224, 224]
            cleans = self.transform(cleans)  # images:[3, 1, 224, 224]
            
            for i in range(images.shape[0]):
                # 先归一化到[0, 1], 相当于Totensor()
                temp_norm = images[i][0]
                temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
                images[i][0] = temp_norm
            # Normalize到[-1, 1]
            images = self.normalize(images)

            for i in range(cleans.shape[0]):
                # 先归一化到[0, 1], 相当于Totensor()
                temp_norm = cleans[i][0]
                temp_norm = (temp_norm - temp_norm.min()) / (temp_norm.max() - temp_norm.min())
                cleans[i][0] = temp_norm
            # Normalize到[-1, 1]
            cleans = self.normalize(cleans)

        if self.random_choose_slice:
            slice_num = random.randint(0, int(images.shape[0])-1)
            images = images[slice_num].unsqueeze(0)
            cleans = cleans[slice_num].unsqueeze(0)
            ddr = ddr[slice_num].unsqueeze(0)
            t60 = t60[slice_num].unsqueeze(0)
            MeanT60 = MeanT60[slice_num].unsqueeze(0)
            valid_info_count = 1
            name_list = [name_list[slice_num]]
        
        if self.channel3:
            images = images.repeat(1, 3, 1, 1)
            cleans = cleans.repeat(1, 3, 1, 1)
        sample = {'image': images, 'ddr': ddr, 't60': t60, "MeanT60": MeanT60, "validlen": valid_info_count,
                  'name': name_list, 'clean': cleans}

        return sample


padding_keys = []
stacking_keys = ['MeanT60', 'ddr', 'image', 't60', 'clean']


def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: [] for k in keys}

    for data in batch:
        for k, v in data.items():
            out[k].append(v)

    for k in stacking_keys:
        # print("key:",k)
        out[k] = torch.cat(out[k], dim=0)  # torch.stack(out[k],dim=0)

    for k in padding_keys:
        out[k] = pad_sequence(out[k], batch_first=True)
    return out
