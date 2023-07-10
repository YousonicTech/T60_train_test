import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Normalize
import random

class Dataset_dict(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=None,root_dir1=None,root_dir2=None,root_dir3=None,root_dir4=None, transform=None, start_freq=0,
                 end_freq=29, which_freq=0, failed_file=".", random_choose_slice=True, channel3=True , train = True ,
                 choose_data = False, choose_key = '_20dB'):
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
        self.dict_root2 = root_dir2
        self.dict_root3 = root_dir3
        self.dict_root4 = root_dir4
        self.transform = transform
        self.dict_data = []
        self.which_freq = which_freq
        self.failed_file = failed_file
        self.normalize = Normalize(mean=[0.5], std=[0.5])
        self.random_choose_slice = random_choose_slice
        self.channel3 = channel3
        self.train = train
        self.choose_data = choose_data
        self.choose_key = choose_key


        for (root, dirs, files) in os.walk(root_dir):
            print("root:", root, "dirs:", dirs, "len of files:", len(files))
            self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
            self.root = root
        if root_dir1 != None:
            for (root, dirs, files) in os.walk(root_dir1):
                print("root:", root, "dirs:", dirs, "len of files:", len(files))
                self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
                self.root = root
        if root_dir2 != None:
            for (root, dirs, files) in os.walk(root_dir2):
                print("root:", root, "dirs:", dirs, "len of files:", len(files))
                self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
                self.root = root
        if root_dir3 != None:
            for (root, dirs, files) in os.walk(root_dir3):
                print("root:", root, "dirs:", dirs, "len of files:", len(files))
                self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
                self.root = root
        if root_dir4 != None:
            for (root, dirs, files) in os.walk(root_dir4):
                print("root:", root, "dirs:", dirs, "len of files:", len(files))
                self.dict_data = self.dict_data + [os.path.join(root, filen) for filen in files]
                self.root = root

        if self.choose_data:
            # self.dict_data = [elem for elem in self.dict_data if self.choose_key in elem]
            self.dict_data = [elem for elem in self.dict_data if not self.choose_key in elem]
            print(len(self.dict_data))




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
            audio_image_dict = torch.load(dict_data_name)
            #print('------------loading------------------ ',dict_data_name)
        except:
            print("failed read file name:", dict_data_name)
            audio_image_dict = torch.load(self.failed_file)
        
        dict_keys = list(audio_image_dict.keys())[0]
        dict_data = audio_image_dict[dict_keys]
        
        
        if  (dict_data[0]['t60'].numel()== 0):
            print('++++++++t60 empty++++++++',dict_keys)
            audio_image_dict = torch.load(self.failed_file)
            dict_keys = list(audio_image_dict.keys())[0]
            dict_data = audio_image_dict[dict_keys]
            
        valid_info_count = 0
        ##下面根据文件名直接读取clean数据
        if self.train:
            if "_TIMIT" in dict_data_name:
                # 以 "_TIMIT" 为标志，将文件名分为两部分
                parts = dict_data_name.split("_TIMIT")
                # 提取往前第二个 "_" 之间的字符,就是clean的名字
                clean_data_name = '/data/hsl/clean_chinese_pt/' + parts[0].split("_")[-2] + '_' + parts[0].split("_")[-1] + '.pt'
            else:
                parts = dict_data_name.split('/')[-1].split('_')[2]
                clean_data_name = '/data/hsl/temp/clean_chinese_pt_gai/'+parts+ '.pt'
        else:
            clean_data_name = "/data/hsl/temp/clean_chinese_pt_gai/粤语女声-6.pt"#测试的话无所谓了，随便传一个值
        clean_data = torch.load(clean_data_name)
        clean_keys = list(clean_data.keys())[0]
        clean_data = clean_data[clean_keys]


        for list_c in range(len(dict_data)):
            if dict_data[list_c] == 0:
                continue
            else:
                valid_info_count += 1
                if "image" in dict_data[list_c].keys():
                    if isinstance(dict_data[list_c]['image'],list):#如果是list，说明是新数据，因为有5个频段

                        temp_image = dict_data[list_c]['image'][self.which_freq]  # 选中对应的倍频程
                        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)
                        
                        if self.train:
                            temp_clean = clean_data[list_c]['image'][self.which_freq]  # 干净语谱图
                            temp_clean = temp_clean[temp_clean.shape[0] // 2:].unsqueeze(0)
                            clean_list.append(temp_clean)

                        images_list.append(temp_image)  # 选图的上半部分，shape=[1, 65, 175] for 2khz, 175是因为只有一个slice(4s)

                        t60_list.append(torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end], 0)) #by hsl
                        ddr_list.append(torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end], 0)) #by hsl
                        MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'], 0))
                        name_list.append(dict_data_name)
                    else:#老数据
                        temp_image = dict_data[list_c]['image']  # 选中对应的倍频程
                        temp_image = temp_image[temp_image.shape[0] // 2:].unsqueeze(0)
                        if self.train:
                            temp_clean = dict_data[list_c]['clean']
                            temp_clean = temp_clean[temp_clean.shape[0] // 2:].unsqueeze(0)
                            clean_list.append(temp_clean)

                        images_list.append(temp_image)  # 选图的上半部分，shape=[1, 65, 175] for 2khz, 175是因为只有一个slice(4s)
                        t60_list.append(
                            torch.unsqueeze(dict_data[list_c]['t60'][self.freq_start:self.freq_end], 0))  # by hsl
                        ddr_list.append(
                            torch.unsqueeze(dict_data[list_c]['ddr'][self.freq_start:self.freq_end], 0))  # by hsl
                        MeanT60_list.append(torch.unsqueeze(dict_data[list_c]['MeanT60'], 0))
                        name_list.append(dict_data_name)




        images = torch.cat(images_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]
        if self.train:
            cleans = torch.cat(clean_list, dim=0).unsqueeze(1)  # [3, 1, 65, 175]
        else:
            cleans = images#这样不会报错，测试用不到clean

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
            ####################meant60改成t60了
        sample = {'image': images, 'ddr': ddr, 't60': t60, "MeanT60": t60, "validlen": valid_info_count,
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
        #print("key:",k)
        #print(out[k])
        out[k] = torch.cat(out[k], dim=0)  # torch.stack(out[k],dim=0)

    for k in padding_keys:
        out[k] = pad_sequence(out[k], batch_first=True)
    return out
