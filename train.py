import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callback import LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    num_classes = 7
    pretrained = False
    input_shape = [256, 256]
    Epoch = 150
    batch_size = 16
    lr = 2e-4
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    model = Unet(num_classes=num_classes)
    if not pretrained:
        weights_init(model)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory("logs/")

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    if True:
        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)


        for epoch in range(Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, dice_loss, focal_loss, cls_weights,
                          num_classes)
            lr_scheduler.step()
