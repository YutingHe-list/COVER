import argparse
import csv
import math
import os
import shutil
import time

import torch.nn.functional as F
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from models.cover import COVER
from utils.STN import SpatialTransformer
from utils.Transform_2d import SpatialTransform2D, CropTransform, AppearanceTransform
from utils.dataloader_SSP_2d import DatasetFromFolder2D as DatasetFromFolder2D_train
from utils.losses import partical_MAE, partical_COS
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-modelname", metavar="NAME", default="COVER_2D_chestxray_ICCV",
                    help="path to dataset")
parser.add_argument("-data", metavar="DIR", default="",
                    help="path to dataset")
parser.add_argument("-j", "--workers", default=16, type=int, metavar="N",
                    help="number of data loading workers (default: 16)")
parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help='number of total epochs to run')
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=24, type=int, metavar="N",
                    help="mini-batch size (default: 24) ")
parser.add_argument("-img-size", default=384, type=int,
                    help="input image size")
parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, metavar="LR",
                    help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                    help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=1, type=int, metavar="N",
                    help="print frequency (default: 10)")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")

parser.add_argument('--n_channels', default=1, type=int,
                    help='number of input channels')
parser.add_argument('--amp', default=2, type=int,
                    help='amplification of the kernal (default: 1)')
parser.add_argument('--degree', default=1.5, type=float,
                    help='spatial transformation degree (default: 1.5)')

def main():
    args = parser.parse_args()

    # Simply call main_worker function
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    model = COVER(n_channels=args.n_channels, dimensions='2D')

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    def count_parameters_in_M(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params / 1e6  # 除以1,000,000转为百万参数

    print(f"model parameter amount：{count_parameters_in_M(model):.2f} M")
    # define loss function (criterion) and optimizer
    criterion_vec = partical_MAE
    criterion_con = partical_COS

    optim_params = model.parameters()

    optimizer = torch.optim.Adam(optim_params, init_lr, weight_decay=args.weight_decay)

    if args.start_epoch > 0:
        if os.path.isfile("checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(args.start_epoch-1)):
            print("=> loading checkpoint '{}'".format("checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(args.start_epoch-1)))
            if args.gpu is None:
                checkpoint = torch.load("checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(args.start_epoch-1))
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load("checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(args.start_epoch-1), map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format("checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(args.start_epoch-1)))


    cudnn.benchmark = True

    # Data loading code
    train_dataset = DatasetFromFolder2D_train(args.data, (int(args.img_size*1.5),)*2)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers
                                               )

    logwriter = LogWriter(name=args.modelname, head=["Batch time", "Data time", "Loss_vec", "Loss_con"])

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        batch_time_log, data_time_log, loss_vec_log, loss_con_log =  train(train_loader, model, criterion_vec, criterion_con, optimizer, epoch, args)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename="checkpoint_"+args.modelname+"_{:04d}.pth.tar".format(epoch),
        )
        logwriter.writeLog([batch_time_log, data_time_log, loss_vec_log, loss_con_log])

def train(train_loader, model, criterion_vec, criterion_con, optimizer, epoch, args):
    batch_time = AverageMeter("Batch time", ":6.3f")
    data_time = AverageMeter("Data time", ":6.3f")
    losses_vec = AverageMeter("Loss_vec", ":.4f")
    losses_con = AverageMeter("Loss_con", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_vec, losses_con],
        prefix="Epoch: [{}]".format(epoch),
    )
    stn = SpatialTransformer()
    app_aug = AppearanceTransform()
    crop_aug = CropTransform((args.img_size, args.img_size))

    degree = args.degree
    spatial_aug = SpatialTransform2D(do_rotation=True,
                                     angle_x=(degree * -np.pi / 9, degree * np.pi / 9),
                                     angle_y=(degree * -np.pi / 9, degree * np.pi / 9),
                                     do_scale=True,
                                     scale_x=(1 - degree * 0.5, 1 + degree * 0.5),
                                     scale_y=(1 - degree * 0.5, 1 + degree * 0.5),
                                     do_translate=True,
                                     trans_x=(degree * -0.2, degree * 0.2),
                                     trans_y=(degree * -0.2, degree * 0.2),
                                     do_shear=True,
                                     shear_xy=(degree * -np.pi / 32, degree * np.pi / 32),
                                     shear_yx=(degree * -np.pi / 32, degree * np.pi / 32),
                                     do_elastic_deform=False,
                                     device="cuda:0" if torch.cuda.is_available() else "cpu")

    # switch to train mode
    model.train()

    end = time.time()
    for idx, (img) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # im_q is fixed image, im_k is moving image

        if args.gpu is not None:
            img = img.cuda(args.gpu, non_blocking=True)

        # Spatial and appearance transformation, crop
        flow_gt = []
        for j in range(img.shape[0]):
            flow, _ = spatial_aug.rand_coords(img.shape[2:])
            flow_gt.append(flow)
        flow_gt = torch.cat(flow_gt, dim=0)
        im_F = stn(img, flow_gt)

        crop_code = crop_aug.rand_code(img.shape[2:])
        im_M = crop_aug.augment_crop(img, crop_code)
        im_M = app_aug.rand_aug(im_M)
        im_F = crop_aug.augment_crop(im_F, crop_code)
        flow_gt = crop_aug.augment_crop(flow_gt, crop_code)

        # Compute output
        flow_pred, f_M, f_F = model(M=im_M, F=im_F)

        # Construct mask
        shape = flow_gt.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow_gt.device)

        new_locs = grid + flow_gt
        for i in range(len(shape)):
            new_locs[:, i, ...] = new_locs[:, i, ...] / (shape[i] - 1)

        new_locs = torch.where(new_locs < 0, torch.full_like(new_locs, 0.), new_locs)
        new_locs = torch.where(new_locs > 1, torch.full_like(new_locs, 0.), new_locs)
        new_locs = torch.where(new_locs > 0, torch.full_like(new_locs, 1.), torch.full_like(new_locs, 0.))
        mask = new_locs[:, 0:1] * new_locs[:, 1:2]

        loss_vec = criterion_vec(flow_pred, flow_gt, mask)
        f_M2F = stn(f_M, flow_gt)

        loss_con = criterion_con(f_F, f_M2F, mask)

        loss = loss_vec + loss_con

        losses_vec.update(loss_vec.item(), img[0].size(0))
        losses_con.update(loss_con.item(), img[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
    return batch_time.avg, data_time.avg, losses_vec.avg, losses_con.avg

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

class LogWriter(object):
    def __init__(self, name, head):
        self.name = name+'.csv'
        with open(self.name, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()

    def writeLog(self, dict):
        with open(self.name, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(dict)
            f.close()

if __name__ == "__main__":
    main()
