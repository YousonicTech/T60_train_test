# T60_train_test
## 汇集了新数据和老数据
## 改写了new_data_load_original.py，可以自动判断不同的数据类型。
## 使用时需要传入两个不同的数据路径:train_dict_root以及train_dict_root1，其中train_dict_root1可以为None
```
train_dict_root = "/data/xbj/0927_500hz_with_clean/train"
train_dict_root1 = "/data2/hsl/0324_pt_with_clean"  # 新数据地址
```
```
train_transformed_dataset = Dataset_dict(root_dir=train_dict_root,root_dir1=train_dict_root1, transform=data_transform,
                                             start_freq=args.start_freq, end_freq=args.end_freq, which_freq=args.which_freq, failed_file=failed_file)

```
## 接着直接运行 `python train_resnet50_500Hz.py`(单卡运行）
# 新增多卡并行训练，`python train_resnet50_500Hz_multi_card.py`
