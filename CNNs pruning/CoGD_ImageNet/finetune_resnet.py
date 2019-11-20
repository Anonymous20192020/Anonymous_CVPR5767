import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import re
import time
import utils.common as utils
from importlib import import_module
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import collections
#from data import cifar10#, cifar100
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.options import args
from utils.preprocess_resnet import prune_resnet
from model import resnet_56_sparse
from utils.utils import get_parameters_size
from resnet import ResNet101
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dataloaders import getTrainValDataset, getTestDataset
import time
import logging
import sys
from ptflops import get_model_complexity_info
import torchvision

num_gpu = 4
batch_size = 256
num_workers=32

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))


save_path='./save'
note = ''
save_path = '{}-{}-{}'.format(save_path, note, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(save_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
kd_flag = 0

def main():
    start_epoch = 0
    best_prec1, best_prec5 = 0.0, 0.0

    ckpt = utils.checkpoint(args)
    writer_train = SummaryWriter(args.job_dir + '/run/train')
    writer_test = SummaryWriter(args.job_dir + '/run/test')

    # Data loading
    print('=> Preparing data..')
    logging.info('=> Preparing data..')

    #loader = import_module('data.' + args.dataset).Data(args)

    # while(1):
    #     a=1

    traindir = os.path.join('/mnt/cephfs_new_wj/cv/ImageNet','ILSVRC2012_img_train')
    valdir = os.path.join('/mnt/cephfs_new_wj/cv/ImageNet','ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_sizes, shuffle=True,
    #     num_workers=8, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    traindir = os.path.join('/mnt/cephfs_new_wj/cv/ImageNet/', 'ILSVRC2012_img_train_rec')
    valdir = os.path.join('/mnt/cephfs_new_wj/cv/ImageNet/', 'ILSVRC2012_img_val_rec')


    train_queue = getTrainValDataset(traindir, valdir, batch_size=batch_size, val_batch_size=batch_size,
                                     num_shards=num_gpu, workers=num_workers)
    valid_queue = getTestDataset(valdir, test_batch_size=batch_size, num_shards=num_gpu,
                                 workers=num_workers)

    #loader = cifar100(args)

    # Create model
    print('=> Building model...')
    logging.info('=> Building model...')
    criterion = nn.CrossEntropyLoss()

    # Fine tune from a checkpoint
    refine = args.refine
    assert refine is not None, 'refine is required'
    checkpoint = torch.load(refine, map_location=torch.device(f"cuda:{args.gpus[0]}"))


    if args.pruned:
        mask = checkpoint['mask']
        model = resnet_56_sparse(has_mask = mask).to(args.gpus[0])
        model.load_state_dict(checkpoint['state_dict_s'])
    else:
        model = prune_resnet(args, checkpoint['state_dict_s'])

    # model = torchvision.models.resnet18()

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
        print('Flops:  ' + flops)
        print('Params: ' + params)
    pruned_dir = args.pruned_dir
    checkpoint_pruned = torch.load(pruned_dir, map_location=torch.device(f"cuda:{args.gpus[0]}"))
    model = torch.nn.DataParallel(model)
    #
    # new_state_dict_pruned = OrderedDict()
    # for k, v in checkpoint_pruned.items():
    #     name = k[7:]
    #     new_state_dict_pruned[name] = v
    # model.load_state_dict(new_state_dict_pruned)

    model.load_state_dict(checkpoint_pruned['state_dict_s'])

    test_prec1, test_prec5 = test(args, valid_queue, model, criterion, writer_test)
    logging.info('Simply test after prune: %e ', test_prec1)
    logging.info('Model size: %e ', get_parameters_size(model)/1e6)

    exit()

    if args.test_only:
        return
    param_s = [param for name, param in model.named_parameters() if 'mask' not in name]
    #optimizer = optim.SGD(model.parameters(), lr=args.lr * 0.00001, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = optim.SGD(param_s, lr=1e-5, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.num_epochs))


    model_kd = None
    if kd_flag:
        model_kd = ResNet101()
        ckpt_kd = torch.load('resnet101.t7', map_location=torch.device(f"cuda:{args.gpus[0]}"))
        state_dict_kd = ckpt_kd['net']
        new_state_dict_kd = OrderedDict()
        for k, v in state_dict_kd.items():
            name = k[7:]
            new_state_dict_kd[name] = v
    #print(new_state_dict_kd)
        model_kd.load_state_dict(new_state_dict_kd)
        model_kd = model_kd.to(args.gpus[1])

    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=torch.device(f"cuda:{args.gpus[0]}"))
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict_s'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))
    #print(model.named_parameters())
    #for name, param in model.named_parameters():
        #print(name)
    for epoch in range(start_epoch, 60):
        scheduler.step()#scheduler.step(epoch)
        t1 = time.time()
        train(args, train_queue, model, criterion, optimizer, writer_train, epoch, model_kd)
        test_prec1, test_prec5 = test(args, valid_queue, model, criterion, writer_test, epoch)
        t2 = time.time()
        print(epoch, t2 - t1)
        logging.info('TEST Top1: %e Top5: %e ', test_prec1, test_prec5)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        print(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")
        logging.info('Best Top1: %e Top5: %e ', best_prec1, best_prec5)

        state = {
            'state_dict_s': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }

        ckpt.save_model(state, epoch + 1, is_best)
        train_queue.reset()
        valid_queue.reset()

    print(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")
    logging.info('Best Top1: %e Top5: %e ', best_prec1, best_prec5)


def train(args, loader_train, model, criterion, optimizer, writer_train, epoch, model_kd = None):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()
    # num_iterations = len(loader_train)

    # for i, (inputs, targets) in enumerate(loader_train, 1):
    for i, data in enumerate(loader_train):
        inputs = torch.cat([data[j]["data"] for j in range(num_gpu)], dim=0)
        targets = torch.cat([data[j]["label"] for j in range(num_gpu)], dim=0).squeeze().long()


        inputs = inputs.to(args.gpus[0])
        targets = targets.to(args.gpus[0])
        logits = model(inputs)
        loss = criterion(logits, targets)

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        if i % 500 == 0:
            print(f'* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
            logging.info('Top1: %e Top5: %e ', top1.avg, top5.avg)

        #kd_flag = 0
        if kd_flag:
            inputs = inputs.to(args.gpus[1])
            features_kd = model_kd(inputs)
            alpha = 0.99
            Temperature = 30
            logits = logits.to(args.gpus[1])
            KD_loss = nn.KLDivLoss()(F.log_softmax(logits / Temperature, dim=1),
                                     F.softmax(features_kd / Temperature, dim=1)) * (
                                  alpha * Temperature * Temperature) #+ F.cross_entropy(logits, targets) * (1 - alpha)
            KD_loss.backward()
        # inputs = inputs.to(args.gpus[0])
        optimizer.step()


def test(args, loader_test, model, criterion, writer_test, epoch=0):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    # num_iterations = len(loader_test)

    print("=> Evaluating...")
    logging.info('=> Evaluating...')
    with torch.no_grad():
        # for i, (inputs, targets) in enumerate(loader_test, 1):
        for i, data in enumerate(loader_test):
            inputs = torch.cat([data[j]["data"] for j in range(num_gpu)], dim=0)
            targets = torch.cat([data[j]["label"] for j in range(num_gpu)], dim=0).squeeze().long()

            inputs = inputs.to(args.gpus[0])
            targets = targets.to(args.gpus[0])

            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            #print(f'* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        print(f'* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        logging.info('Top1: %e Top5: %e ', top1.avg, top5.avg)

    if not args.test_only:
        writer_test.add_scalar('test_top1', top1.avg, epoch)

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()