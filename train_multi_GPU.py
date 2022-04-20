import os
import math
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit_multi import fit_one_epoch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings('ignore')

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main(args):

    local_rank = args.local_rank
    
    if local_rank == 0:
        print(args)
    # DDP：DDP backend初始化
    Cuda = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    num_classes = args.num_classes
    pretrained = args.backbone_pretrained
    input_shape = args.input_shape
    num_epoch = args.num_epoch
    batch_size = int(args.batch_size/dist.get_world_size())
    lr = args.lr
    VOCdevkit_path = args.data_set
    warmup_epoch = args.warmup_epoch
    dice_loss = args.dice_loss
    focal_loss = args.focal_loss
    #cls_weights = np.array([1, 1, 1, 2, 1, 2, 1], np.float32)
    cls_weights = np.ones([num_classes], np.float32)
    Freeze_Train = args.Freeze_Train
    num_workers = args.num_workers

    model = Unet(num_classes=num_classes)
    if not pretrained:
        weights_init(model)

    model_path = args.model_path
    if model_path is not None:
        print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location='cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        if local_rank == 0:
            torch.save(model.state_dict(), "initial_weights.pt")
        dist.barrier()
        model.load_state_dict(torch.load("initial_weights.pt", map_location='cpu'))

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('use sync-bn')

    model = model.to(local_rank)

    if Cuda:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True if Freeze_Train else False)
        model_train = model.train()
        cudnn.benchmark = True

    loss_history = LossHistory("logs/", args)

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    gen = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    optimizer = optim.Adam(model_train.parameters(), lr)

    iter_per_epoch = len(train_dataset)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)

    if args.lrf is not None:
        lf = lambda x: ((1 + math.cos(x * math.pi / args.num_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)

    if Freeze_Train:
        model.module.freeze_backbone()

    for epoch in range(0, num_epoch):
        gen.sampler.set_epoch(epoch)
        gen_val.sampler.set_epoch(epoch)

        fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      num_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, args, scaler, warmup_scheduler, warmup_epoch)

        if epoch >= warmup_epoch:
            lr_scheduler.step()
        if Freeze_Train and epoch >= args.Freeze_epoch:
            model.module.unfreeze_backbone()

    if local_rank == 0:
        if os.path.exists("initial_weights.pt"):
            os.remove("initial_weights.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int, help="Don't change it")
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--input_shape", default=[256, 256], type=list, help='image size for CNN after Data Augmentation')
    parser.add_argument("--lr", default=0.0001, type=float, help='learn rate')
    parser.add_argument("--lrf", default=0.000001, type=float, help='if it is not None, use CosineAnnealing, else use lr_step')
    parser.add_argument("--warmup_epoch", default=5, type=int, help='num epoch for warm up')
    parser.add_argument("--num_classes", default=7, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help='total batch_size')
    parser.add_argument("--accumulation_steps", default=None, type=int, help='if not None, use Gradient accumulation')
    parser.add_argument("--backbone_pretrained", default=False, type=bool, help='pretrained model of backbone')
    parser.add_argument("--Freeze_Train", default=False, type=bool, help='if backbone_pretrained, suggest to use this')
    parser.add_argument("--Freeze_epoch", default=None, type=int, help='if Freeze_Train, use this, num epoch of freeze the backbone to train')
    parser.add_argument("--num_epoch", default=200, type=int, help='total epoch to train')
    parser.add_argument("--model_path", default=None, type=str, help='pretrained model of all the net, if not None, use it')
    parser.add_argument("--data_set", default='VOCdevkit', type=str)
    parser.add_argument("--sync_bn", default=True, type=bool, help='use sync_bn')
    parser.add_argument("--dice_loss", default=True, type=bool, help='if num_classes<10, use this')
    parser.add_argument("--focal_loss", default=False, type=bool, help='if all kinds of samples is unbalanced, use this, else ce_loss is default set')
    args = parser.parse_args()
    main(args)
