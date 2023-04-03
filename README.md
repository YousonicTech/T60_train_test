# T60_train_test
# -------------训练部分------------
## 改写了new_data_load_original.py，可以自动判断不同的数据类型。
即老数据中的image为单个频段，新数据中image为多个频段，可以自动判断并读取。
使用时需要传入两个不同的数据路径:train_dict_root以及train_dict_root1，其中train_dict_root1可以为None
```
train_dict_root = "/data/xbj/0927_500hz_with_clean/train"
train_dict_root1 = "/data2/hsl/0324_pt_with_clean"  # 新数据地址
```
```
train_transformed_dataset = Dataset_dict(root_dir=train_dict_root,root_dir1=train_dict_root1, transform=data_transform,
                                             start_freq=args.start_freq, end_freq=args.end_freq, which_freq=args.which_freq, failed_file=failed_file)

```
## 接着直接运行 `python train_resnet50_500Hz.py`(单卡运行）
## 多卡并行训练，单进程、多线程
`python train_resnet50_500Hz_multi_card.py` 
## 多卡并行训练，多进程、多线程（更推荐）
`python -m torch.distributed.launch --nnodes=1 --nproc_per_node=3 train_resnet50_1000Hz_multi_card_DDP.py`
其中nnodes为机器数，不用改，nproc_per_node为显卡数，更快更推荐，使用时记得更改数据集路径及选择频率。
```
    parser.add_argument('--which_freq', type=int, default=2)   # 记得改成对应的图编号
    parser.add_argument('--freq_loc', type=int, default=13)    # 记得改成对应的倍频程
```

# -------------测试部分------------
# T60_train_test/test_resnet50_500Hz_old  
## Code Description  
### Main documents for testing  
test_resnet50_500Hz.py  
  
  The *.py is used to test previous data, and it can test different frequency. You can run your own test by the Run Instructions.
  
### Run Instructions   
If you want to test the newly generated data, you should revise the following place:  
  1、The default of the following two codes should be modified to the path of your pretrained model.

    parser.add_argument('--save_dir', type=str, default="/data2/pyq/yousonic/old_500hz/1001_autoLoss/save_model/")  
    parser.add_argument('--model_path', type=str, default="/data2/pyq/yousonic/old_500hz/1001_autoLoss/save_model/")  
  
  2、The default represents the number of epochs.  
  
    parser.add_argument('--epoch_for_save', type=int, default=60)  
    
  3、The defaults are used to choose frequency, and the corresponding relationships are  125Hz:(0, 7); 250Hz:(1, 10); 500Hz:(2, 13); 1kHz:(3, 16); 2kHz:(4, 19).  
  
    parser.add_argument('--which_freq', type=int, default=2)   # 记得改成对应的图编号
    parser.add_argument('--freq_loc', type=int, default=13)    # 记得改成对应的倍频程
    
   4、The default should be modified to your own save-path.  
   
    parser.add_argument('--outputresult_dir', type=str, default="./0329_oldmodel_yqhdata_500hz/epoch60")  
    
   5、The val_dict_root and failed file should be modified to your own data path and you can revise the val_batch_size. However your data must have clean !!!
   
    if DEBUG == False:
        val_dict_root = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset"
        val_batch_size = 1
        failed_file = "/Users/bajianxiang/Desktop/internship/filter_down_spec_dataset/koli-national-park-winter/koli-national-park-winter_koli_snow_site4_1way_bformat_1_koli-national-park-winter_东北话男声_1_TIMIT_a005_100_110_10dB-0.pt"
    else:
        val_dict_root = "/data/xbj/0927_500hz_with_clean/val"
        failed_file = "/data/xbj/0927_500hz_with_clean/val/hoffmann-lime-kiln-langcliffeuk/hoffmann-lime-kiln-langcliffeuk_ir_p2_0_1_hoffmann-lime-kiln-langcliffeuk_东北话男声_1_TIMIT_a053_110_120_10dB-0.pt"
        val_batch_size = 6  
   
   6、If you use model train through multi-card, you should add the following code.
   
       net = nn.DataParallel(net)
   
### Files that do not need to be changed！！！
model  
utils  
data_load.py  
data_load_aug2.py  
data_load_for_FPN_spec.py  
eval_same_room.py  
gen_boxplot.py  
get_deverb_spec_test.py  
gtg.py  
load_same_room.py  
new_data_load_original.py
new_data_load_original_for_our_data.py  
new_data_load_origin_old.py  
new_thread_test.py  
splweighting.py  
SSIMLoss.py  
Test_new_dataset.py  
test_our_data.py  
thread_test.py  
train_resnet50_500Hz.py  
valdata_meant60.py  
workspace_utils.py  
