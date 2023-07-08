# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm, trange
import math
import torchvision.transforms


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')
import valdata_meant60
from valdata_meant60 import Val_meanT60
from torch.utils.tensorboard import SummaryWriter
# 决定使用哪块GPU进行训练
import csv
import os
import glob
from new_data_load_original import Dataset_dict, collate_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device_ids = [0, 1]
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
from model.attentionFPN import FPN

torch.backends.cudnn.benchmark = True

#TODO: 修改model_path,test_epochs,test_freq,outputresult_dir_head(这个可以不用动）,ln_out,freq_loc
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the mmTransformer')
    parser.add_argument('--model_path', type=str,
                        default="/data2/hsl/T60_train_test/save_model/0707_FULLBAND3_attFPN_old_hybl_zgc_yqh_zgcfx_NONoise_3040_1_2_random/",#
                        help= '保存的模型路径')
    parser.add_argument('--test_epochs', type=list, default=[10,31,61],help='要测试的epoch，列表')  # 记得改epoch数
    #parser.add_argument('--test_freq', type=int, default=2000,choices=[500,1000,2000,4000])#这个不需要了，换成直接指定freq_loc
    parser.add_argument('--start_freq', type=int, default=0)
    parser.add_argument('--end_freq', type=int, default=29)
    parser.add_argument('--ln_out', type=int, default=3)
    parser.add_argument('--freq_loc', type=int, default=[13,16, 19])  # [13,16,19]
    parser.add_argument('--outputresult_dir_head', type=str,default="/data2/hsl/T60_train_test/test_result/0708test_Fullband3/",help='输出测试结果的路径')
    args = parser.parse_args()
    return args
#TODO:修改val_dict_root_list,failed_file_list

# val_dict_root_list=["/data3/hsl/TEST_PT/LiveRecord/val_ZGCFX_LiveRecord/",
#                     "/data3/hsl/TEST_PT/LiveRecord/val_YQH_LiveRecord/",
#                     "/data3/hsl/TEST_PT/LiveRecord/val_hybl_LiveRecord/",
#                     "/data3/hsl/0515_oldval_pt_with_clean/"
#                     ]
# failed_file_list =["/data3/hsl/TEST_PT/LiveRecord/val_ZGCFX_LiveRecord/ZGCFX-DGNT/ZGCFX-DGNT-ch5-0.pt",
#                     "/data3/hsl/TEST_PT/LiveRecord/val_hybl_LiveRecord/five-two/five-two-ch3-0.pt",
#                    "/data3/hsl/TEST_PT/LiveRecord/val_hybl_LiveRecord/five-two/five-two-ch3-0.pt",
#                     "/data3/hsl/0515_oldval_pt_with_clean/koli-national-park-summer/koli-national-park-summer_koli_summer_site1_1way_bformat_4_koli-national-park-summer_粤语女声_6_TIMIT_b037_160_170_10dB-0.pt"
#                    ]
# "/data3/hsl/0515_oldval_pt_with_clean/"
# "/data3/hsl/0515_oldval_pt_with_clean/koli-national-park-summer/koli-national-park-summer_koli_summer_site1_1way_bformat_4_koli-national-park-summer_粤语女声_6_TIMIT_b037_160_170_10dB-0.pt"

# val_dict_root_list=["/data3/hsl/TEST_PT/LiveRecord_Split/val_ZGCFX_LiveRecord/",
#                     "/data3/hsl/TEST_PT/LiveRecord_Split/val_hybl_LiveRecord_Split/",
#                     "/data3/hsl/TEST_PT/LiveRecord_Split/val_YQH_LiveRecord_Split/",
#
#                     ]
# failed_file_list =["/data3/hsl/TEST_PT/LiveRecord_Split/val_ZGCFX_LiveRecord/ZGCFX-1-3/ZGCFX-1-3-ch2-0.pt",
#                     "/data3/hsl/TEST_PT/LiveRecord_Split/val_hybl_LiveRecord_Split/eight-two/eight-two-ch2-0_0.pt",
#                    "/data3/hsl/TEST_PT/LiveRecord_Split/val_YQH_LiveRecord_Split/YQH101/YQH101-ch1-0_0.pt",
#                                       ]

val_dict_root_list=[
                    "/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/",
                    "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/"
                    ]
failed_file_list =["/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/ZGCFX-1-3/ZGCFX-1-3-ch4-0_25.pt",
                    "/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/eight-two/eight-two-ch2-0_25.pt",
                   "/data3/hsl/TEST_PT/500/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/YQH101/YQH101-ch1-0_25.pt",
                    "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/ZGCFX-1-3/ZGCFX-1-3-ch4-0_25.pt",
                    "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/eight-two/eight-two-ch2-0_25.pt",
                   "/data3/hsl/TEST_PT/1K/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/YQH101/YQH101-ch1-0_25.pt",
                    "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_ZGCFX_LiveRecord_QTQW/ZGCFX-1-3/ZGCFX-1-3-ch4-0_25.pt",
                    "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_hybl_LiveRecord_QTQW/eight-two/eight-two-ch2-0_25.pt",
                   "/data3/hsl/TEST_PT/2K/LiveRecord_QTQW/val_YQH_LiveRecord_QTQW/YQH101/YQH101-ch1-0_25.pt"
                    ]


# def get_freq_index(test_freq):不需要了
#     # 125Hz:(0, 7); 250Hz:(1, 10); 500Hz:(2, 13); 1kHz:(3, 16); 2kHz:(4, 19)
#     if test_freq == 500:
#         which_freq = 2
#         freq_loc = 13
#     elif test_freq == 1000:
#         which_freq = 3
#         freq_loc = 16
#     elif test_freq == 2000:
#         which_freq = 4
#         freq_loc = 19
#     elif test_freq == 4000:
#         which_freq = 4
#         freq_loc = 19
#     return which_freq,freq_loc


def load_checkpoint(checkpoint_path=None, model=None, device=None):
    save_model = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(save_model['model'])
    return model


def net_sample_output():
    for i, sample in enumerate(val_loader):
        images = sample['image']
        ddr = sample['ddr']
        t60 = sample['t60']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = torch.tensor(images.clone().detach(), dtype=torch.float32, device=device)

        # forward pass to get net output
        t60_predict = net(images)

        # break after first image is tested
        if i == 0:
            return images, t60_predict, t60


def val_net(net, epoch, val_loader, writer):
    with torch.no_grad():
        net.eval()
        total_mean_loss = 0
        total_mean_bias = 0
        progress_bar = tqdm(val_loader)
        for j, datas in enumerate(progress_bar):
            images_reshape = datas['image'].to(torch.float32).to(device)  # 服务器上这个to(device)改了，得在CPU上运算
            valid_len = datas['validlen']
            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)

            gt_t60_reshape = gt_t60_reshape.T[freq_loc, :, :].T  # [total_slices, 1]
            output_pts = net(images_reshape, valid_len)

            loss = criterion(output_pts, gt_t60_reshape)
            bias = torch.sum((gt_t60_reshape - output_pts)) / output_pts.shape[0]

            total_mean_loss += loss.item()
            total_mean_bias += bias.item()

        mean_loss = total_mean_loss / len(val_loader)
        mean_bias = float(total_mean_bias) / len(val_loader)
        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" % (epoch, mean_loss, mean_bias))


def output_result_analysis(result_dict, output_dir, test_path):
    test_dataset = test_path.split("/")[-2]
    if not test_dataset:
        test_dataset = "hahahah"
    csv_file = os.path.join(output_dir, test_dataset + ".csv")

    f = open(csv_file, "w")
    csv_writer_normal = csv.writer(f)
    # bias_lst = []
    for key, value in result_dict.items():
        freq_name_list=[0]
        for feq in args.freq_loc:
            if feq == 13:
                freq_name_list.append('500Hz')
            elif feq == 16:
                freq_name_list.append('1KHz')
            elif feq == 19:
                freq_name_list.append('2KHz')
            elif feq == 22:
                freq_name_list.append('4KHz')

        # freq_name = 125 * 2 ** which_freq
        # if freq_name < 1000:
        #     freq_name = str(freq_name) + 'Hz'
        # else:
        #     freq_name = str(int(freq_name / 1000)) + 'k Hz'
        csv_writer = csv_writer_normal
        csv_writer.writerow(freq_name_list)
        csv_writer.writerow([str(key)])

        for k in range(len(result_dict[key]["output_list"])):
            csv_writer.writerow(["output%d" % k] + result_dict[key]["output_list"][k].cpu().numpy().tolist())
        csv_writer.writerow(["mean_output"] + result_dict[key]["mean_output"].cpu().numpy().tolist())
        csv_writer.writerow(["gt"] + result_dict[key]["gt"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_bias"] + result_dict[key]["mean_bias"].cpu().numpy().tolist())
        # bias_lst.extend(result_dict[key]["mean_bias"].cpu().numpy().tolist())
        csv_writer.writerow(["mean_mse"] + (result_dict[key]["mean_bias"] ** 2).cpu().numpy().tolist())
        csv_writer.writerow([" "])
        csv_writer.writerow([" "])


def test_net(net, epoch, val_loader, writer, batch_size):
    result_dict = {}
    with torch.no_grad():
        net.eval()

        total_mean_loss = torch.zeros((1, args.ln_out))
        total_mean_bias = torch.zeros((1, args.ln_out))

        progress_bar = tqdm(val_loader)
        useless_count = 0
        for j, datas in enumerate(progress_bar):

            images_reshape = datas['image'].to(torch.float32).to(device)
            #print(images_reshape.shape)

            gt_t60_reshape = datas['t60'].to(torch.float32).to(device).unsqueeze(1)

            valid_len = datas['validlen']
            #print(valid_len)
            names = datas['name']
            # gt_t60_reshape会发生变化
            names_keys = str(names[0]).split('/')[-1]
            names_keys = names_keys.split('.')[0]

            gt_t60_reshape = gt_t60_reshape.T[freq_loc, :, :].T  # [total_slices, 1]
            gt_t60_reshape = gt_t60_reshape.squeeze() #这里和单频段cehi不一样，多加这一行
            output_pts, _ = net(images_reshape, valid_len)

            bias = gt_t60_reshape - output_pts
            rsquare_error = torch.sqrt(bias ** 2)
            abs_bias = torch.abs(gt_t60_reshape - output_pts)

            if not torch.isnan(rsquare_error).all():
                total_mean_loss += torch.mean(torch.mean(rsquare_error, dim=0), dim=0).cpu().detach()
                total_mean_bias += torch.mean(torch.mean(bias, dim=0), dim=0).cpu().detach()

            else:
                useless_count += 1

            for i in range(len(valid_len)):
                start_num = 0
                if i > 0:
                    start_num = sum(valid_len[0:i])

                output_list = [output_pts[k] for k in range(start_num, start_num + valid_len[i])]
                mean_output = torch.mean(output_pts[start_num:start_num + valid_len[i]], dim=0)
                mean_abs_bias = torch.mean(abs_bias[start_num:start_num + valid_len[i]], dim=0)
                mean_bias = torch.mean(bias[start_num:start_num + valid_len[i]], dim=0)
                mean_rsquare_error = torch.mean(rsquare_error[start_num:start_num + valid_len[i]], dim=0)
                # mean_gt = torch.mean(gt_t60_reshape[0:0], dim=0)
                mean_gt = torch.mean(gt_t60_reshape[start_num:start_num + valid_len[i]], dim=0)
                result_dict[names[i][0]] = {"mean_output": mean_output, "mean_bias": mean_bias, "gt": mean_gt,
                                            "square_error": mean_rsquare_error, "output_list": output_list}

        mean_loss = total_mean_loss / (len(val_loader) - batch_size * useless_count)
        mean_bias = total_mean_bias / (len(val_loader) - batch_size * useless_count)

        if not writer is None:
            writer.add_scalar('val/mean_loss', mean_loss, epoch)
            writer.add_scalar('val/mean_bias', mean_bias, epoch)

        print("epoch:", epoch, "Mean loss:", mean_loss)
        print("epoch:", epoch, "Mean bias:", mean_bias)

        # print("Epoch %d Evaluation: mean loss:%f,mean bias:%f" %(epoch,mean_loss,mean_bias))

        return result_dict


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.warmup = False
        self.warmup_epochs = 10
        self.min_lr = min(lr)
        self.max_lr = max(lr)
        # self.lr = lr
        # self.lr_epochs = lr_epochs

        # self.lr_warmup = lambda epoch_in : min(self.lr)+0.5*(max(self.lr) - min(self.lr))*(1+np.cos((epoch_in-self.warmup_epochs)*PI/(2*self.warmup_epochs)))
        self.lr_warmup = lambda epoch_in: self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos((epoch_in - self.warmup_epochs) * math.PI / (2 * self.warmup_epochs)))
        if self.warmup == True:
            self.lr_epochs = [self.warmup_epochs] + [i + self.warmup_epochs for i in lr_epochs]
            self.lr = [self.lr_warmup] + lr
        else:
            self.lr_epochs = lr_epochs
            self.lr = lr

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        if self.warmup == True and idx == 0:
            return self.lr[idx](epoch)
        else:
            return self.lr[idx]


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=20)


def get_boxplot(test_res):
    room_dict = {}

    for key, value in test_res.items():
        if '.pt' not in key:
            continue
        room_config, room_name = key.split('/')[-2:]
        room = room_name.split('.')[0]  # 'N208-first-dot-0'
        # ----------yanqihu&zhongguancun---------------------- #
        # temp_room = room_name.split('-')[0]  # 'N208'
        # -------------val---------------------- #
        temp_room = room_name.split(room_config)[1]
        temp = np.array(torch.load(key)[room][0]['t60'][freq_loc].unsqueeze(0)).tolist()

        if temp_room not in room_dict.keys():
            # then create a new room_dict
            value['gt'] = temp * 6
            room_dict[temp_room] = {'gt': value['gt'], 'mean_output': value['mean_output'].tolist(),
                                    'mean_bias': value['mean_bias'].tolist()}
        else:
            value['gt'] = temp * 6
            # print(room_name, value['gt'])
            room_dict[temp_room]['gt'].extend(value['gt'])
            # room_dict[temp_room]['gt'].extend(temp)
            room_dict[temp_room]['mean_output'].extend(value['mean_output'].tolist())
            room_dict[temp_room]['mean_bias'].extend(value['mean_bias'].tolist())
    for room_key in room_dict.keys():
        room_dict[room_key]['gt'] = [sum(np.array(room_dict[room_key]['gt'])) / len(room_dict[room_key]['gt'])] * 12

    """"aaa"""
    fig = plt.figure(figsize=(60, 30))
    ticks = list(room_dict.keys())
    for i in range(len(ticks)):
        if i // 2 == 0:
            ticks[i] = '\n' + ticks[i]
    gt_plot = plt.boxplot([v['gt'] for _, v in room_dict.items()],
                          positions=np.array(np.arange(len(room_dict.values()))) * 2.0 - 0.5, widths=0.5)
    mean_output_plot = plt.boxplot([v['mean_output'] for _, v in room_dict.items()],
                                   positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.5, widths=0.5)
    # mean_bias_plot = plt.boxplot([v['mean_bias'] for _, v in room_dict.items()],
    #                             positions=np.array(np.arange(len(room_dict.values()))) * 2.0 + 0.6, widths=0.3)

    # setting colors for each groups
    define_box_properties(gt_plot, 'black', 'gt')
    define_box_properties(mean_output_plot, '#D7191C', 'mean_output')
    # define_box_properties(mean_bias_plot, '#2C7BB6', 'mean_bias')

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks, fontsize=20, rotation=30)
    plt.yticks(fontsize=20)
    # plt.tight_layout()
    # set the title
    # plt.xticks(rotation=-45)

    fig_name = 'Test_' + freq_name + '_epoch' + str(args.epoch_for_save)
    plt.title(fig_name, fontsize=30)  # 标题，并设定字号大小
    plt.show()
    plt.savefig(outputresult_dir + '/' + fig_name + '.png', dpi=fig.dpi, pad_inches=4)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}

# val_dict_root = "/data3/hsl/TEST_PT/PV_OldData_066_test/"
# failed_file = "/data3/hsl/TEST_PT/PV_OldData_066_test/ron-cooke-hub-university-york/ron-cooke-hub-university-york_tstr-ir-3_粤语女声-3_N_NdB-0.pt"
#
# failed_file =
#
# failed_file =
# val_dict_root =
# failed_file =
#
# failed_file = "/data3/hsl/TEST_PT/LiveRecord_Split/val_ZGCFX_LiveRecord_Split/ZGCFX-1-1/ZGCFX-1-1-ch1-0_10.pt"
# val_dict_root = "/data3/hsl/0515_oldval_pt_with_clean/"
# failed_file = "/data3/hsl/0515_oldval_pt_with_clean/koli-national-park-summer/koli-national-park-summer_koli_summer_site1_1way_bformat_4_koli-national-park-summer_粤语女声_6_TIMIT_b055_50_60_0dB-0.pt"

if __name__ == "__main__":

    args = parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # which_freq, freq_loc =get_freq_index(args.test_freq)
    which_freq = 0#随便复制，dataloader里会自动识别修改
    freq_loc = args.freq_loc
    for test_epoch in args.test_epochs:
        model_path = args.model_path + 't60_predict_model_' + str(test_epoch) + '_fullT60_rir_timit_noise.pt'
        net = FPN(num_blocks=[2, 4, 23, 3], num_classes=3, back_bone="resnet50", pretrained=False,ln_out=args.ln_out)
        start_time = time.time()
        net = load_checkpoint(model_path, net, device)
        print('Successfully Loaded model: {}'.format(model_path))
        print('Finished Initialization in {:.3f}s!!!'.format(time.time() - start_time))
        net.to(device)
        print(get_parameter_number(net))


        criterion = torch.nn.MSELoss()
        data_transform = transforms.Compose([transforms.Resize([224, 224])])
        val_batch_size = 10
        for i in range(len(val_dict_root_list)):
            val_dict_root = val_dict_root_list[i]
            failed_file = failed_file_list[i]
            # if val_dict_root.split('/')[-2] == '0515_oldval_pt_with_clean':
            #     choose_data = True
            # else:
            #     choose_data = False
            choose_data = True
            val_transformed_dataset = Dataset_dict(root_dir=val_dict_root,
                                                   transform=data_transform,
                                                   start_freq=args.start_freq, end_freq=args.end_freq,
                                                   which_freq=which_freq,
                                                   failed_file=failed_file, random_choose_slice=False, train=False)
            print("len of val dataset:", len(val_transformed_dataset))
            val_loader = torch.utils.data.DataLoader(val_transformed_dataset,
                                                     shuffle=False, num_workers=4,
                                                     batch_size=val_batch_size, drop_last=False,
                                                     collate_fn=collate_fn)
            print("after test loader init")
            test_output_result = test_net(net, test_epoch, val_loader, None, val_batch_size)
            # outputresult_dir = args.outputresult_dir_head + args.model_path.split('/')[-2] +'/'+ val_dict_root.split('/')[-4] \
            #                    +'/' + val_dict_root.split('/')[-2]+ '/' + 'epoch' + str(test_epoch)
            outputresult_dir = args.outputresult_dir_head + args.model_path.split('/')[-2] +'/'+ '/' + val_dict_root.split('/')[-2]+ \
                               '/' + 'epoch' + str(test_epoch)
            if not os.path.exists(outputresult_dir):
                os.makedirs(outputresult_dir)

            output_result_analysis(test_output_result, outputresult_dir, val_dict_root)
            save_path = outputresult_dir + '/'  + str(test_epoch) + '.pt'
            torch.save(test_output_result, save_path)

