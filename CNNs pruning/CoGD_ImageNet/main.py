import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import utils.common as utils
from utils.options import args
from utils.preprocess import prune_resnet

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

from fista import FISTA
from model import Discriminator, resnet_56, resnet_56_sparse
from data import cifar10

from resnet import ResNet18, ResNet50
from resnet_sprase import ResNet18_sprase,ResNet50_sprase

from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from resnet_imagenet import resnet101

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import utils.globalvar as gl
gl._init()
import time
import logging
import sys

try:
    import nvidia.dali.plugin.pytorch as plugin_pytorch
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


num_gpu = 4
batch_sizes = 256
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

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, num_shards, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path=[os.path.join(data_dir, "_train.rec")],
                                     index_path=[os.path.join(data_dir, "_train.idx")], random_shuffle=True,
                                     shard_id=device_id, num_shards=num_shards)

        # self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=4, random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.rrc = ops.RandomResizedCrop(device=dali_device, size=(crop, crop))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, num_shards, dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path=[os.path.join(data_dir, "_val.rec")],
                                     index_path=[os.path.join(data_dir, "_val.idx")],
                                     random_shuffle=False, shard_id=device_id, num_shards=num_shards)

        # self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=4, random_shuffle=False)

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        # self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
def getTrainValDataset(traindir, valdir, batch_size, test_batch_size, num_shards, workers):
    pipes = [
        HybridTrainPipe(batch_size=int(batch_size / num_shards), num_threads=workers,
                             device_id=device_id,
                             data_dir=traindir, crop=224, num_shards=num_shards) for device_id in
        range(num_shards)]
    pipes[0].build()
    train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

    pipes = [
        HybridValPipe(batch_size=int(test_batch_size / num_shards), num_threads=num_gpu, device_id=device_id,
                           data_dir=valdir,
                           crop=224, size=256, num_shards=num_shards) for device_id in range(num_shards)]
    pipes[0].build()
    val_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))
    return train_loader, val_loader

def cal_model_flops_params(model):
    model.eval()
    input = torch.ones([1, 3, 224, 224], dtype = torch.float32).cuda()
    flops_list=[]
    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        flops = (self.out_channels/self.groups) * (self.kernel_size[0] * self.kernel_size[1] *self.in_channels/self.groups) * output_height * output_width*self.groups
        flops_list.append(flops)
        print(flops, input[0].shape, output[0].shape)

    def linear_hook(self, input, output):
        flops = self.in_features*self.out_features
        flops_list.append(flops)
        print(flops)
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            foo(c)


def main():
    checkpoint = utils.checkpoint(args)
    writer_train = SummaryWriter(args.job_dir + '/run/train')
    writer_test = SummaryWriter(args.job_dir + '/run/test')

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0
    
    # Data loading
    # while(1):
    #     a=2
    print('=> Preparing data..')
    logging.info('=> Preparing data..')

    traindir = os.path.join('/mnt/cephfs_hl/cv/ImageNet/', 'ILSVRC2012_img_train_rec')
    valdir = os.path.join('/mnt/cephfs_hl/cv/ImageNet/', 'ILSVRC2012_img_val_rec')
    train_loader, val_loader = getTrainValDataset(traindir, valdir, batch_sizes, 100, num_gpu, num_workers)

    # Create model
    print('=> Building model...')
    logging.info('=> Building model...')


    model_t = ResNet50()

    # model_kd = resnet101(pretrained=False)
   
    #print(model_kd)
    # Load teacher model
    ckpt_t = torch.load(args.teacher_dir, map_location=torch.device(f"cuda:{args.gpus[0]}"))
    state_dict_t = ckpt_t
    new_state_dict_t = OrderedDict()

    new_state_dict_t = state_dict_t

    model_t.load_state_dict(new_state_dict_t)
    model_t = model_t.to(args.gpus[0])

    for para in list(model_t.parameters())[:-2]:
        para.requires_grad = False

    model_s = ResNet50_sprase().to(args.gpus[0])
    model_dict_s = model_s.state_dict()
    model_dict_s.update(new_state_dict_t)
    model_s.load_state_dict(model_dict_s)


    #ckpt_kd = torch.load('resnet101-5d3b4d8f.pth', map_location=torch.device(f"cuda:{args.gpus[0]}"))
    #state_dict_kd = ckpt_kd
    #new_state_dict_kd = state_dict_kd
    #model_kd.load_state_dict(new_state_dict_kd)
    #model_kd = model_kd.to(args.gpus[0])

    #for para in list(model_kd.parameters())[:-2]:
        #para.requires_grad = False

    model_d = Discriminator().to(args.gpus[0])

    model_s = nn.DataParallel(model_s).cuda()
    model_t = nn.DataParallel(model_t).cuda()
    model_d = nn.DataParallel(model_d).cuda()


    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_m = FISTA(param_m, lr=args.lr * 100, gamma=args.sparse_lambda)

    scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=torch.device(f"cuda:{args.gpus[0]}"))
        state_dict_s = ckpt['state_dict_s']
        state_dict_d = ckpt['state_dict_d']


        new_state_dict_s = OrderedDict()
        for k, v in state_dict_s.items():
            new_state_dict_s['module.'+k] = v

        best_prec1 = ckpt['best_prec1']
        model_s.load_state_dict(new_state_dict_s)
        model_d.load_state_dict(ckpt['state_dict_d'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        start_epoch = ckpt['epoch']
        print('=> Continue from epoch {}...'.format(ckpt['epoch']))

    models = [model_t, model_s, model_d]#, model_kd]
    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]

    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)

        #global g_e
        #g_e = epoch
        #gl.set_value('epoch',g_e)      

        train(args, train_loader, models, optimizers, epoch, writer_train)
        test_prec1, test_prec5 = test(args, val_loader, model_s)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()

        state = {
            'state_dict_s': model_state_dict,
            'state_dict_d': model_d.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_s': optimizer_s.state_dict(),
            'optimizer_m': optimizer_m.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'scheduler_m': scheduler_m.state_dict(),
            'epoch': epoch + 1
        }
        train_loader.reset()
        val_loader.reset()
        #if is_best:
        checkpoint.save_model(state, epoch + 1, is_best)
        #checkpoint.save_model(state, 1, False)

    print(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")
    logging.info('Best Top1: %e Top5: %e ', best_prec1, best_prec5)

    # best_model = torch.load(f'{args.job_dir}/checkpoint/model_best.pt', map_location=torch.device(f"cuda:{args.gpus[0]}"))
    # model = prune_resnet(args, best_model['state_dict_s'])


def train(args, loader_train, models, optimizers, epoch, writer_train):
    #losses_d = utils.AverageMeter()
    #losses_data = utils.AverageMeter()
    #losses_g = utils.AverageMeter()
    #losses_sparse = utils.AverageMeter()
    #losses_kl = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
    model_s = models[1]
    model_d = models[2]
    #model_kd = models[3]

    bce_logits = nn.BCEWithLogitsLoss()

    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]
    optimizer_m = optimizers[2]

    # switch to train mode
    model_d.train()
    model_s.train()
    num_iterations = int(loader_train._size / batch_sizes)
    #num_iterations = len(loader_train)
    print(num_iterations)
    real_label = 1
    fake_label = 0    
    exact_list=["layer3"]
    num_pruned = -1
    t0 = time.time()

    '''
    prec1 = [60]
    #prec1 = 0
    error_d = 0
    error_sparse = 0
    error_g = 0
    error_data = 0
    KD_loss = 0

    alpha_d = args.miu * ( 0.9 - epoch / args.num_epochs * 0.9 )
    sparse_lambda = args.sparse_lambda
    mask_step = args.mask_step
    lr_decay_step = args.lr_decay_step
    '''
    #for i, (inputs, targets) in enumerate(loader_train, 1):     
    for i, data in enumerate(loader_train): 


        global iteration
        iteration = i

        tt0 = time.time()
        if i % 60 == 1:
            t0 = time.time()

        if i % 400 == 1:
            num_mask = []
            for name, weight in model_s.named_parameters():
                if 'mask' in name:
                    for ii in range(len(weight)):
                        num_mask.append(weight[ii].item())
            num_pruned = sum(m == 0 for m in num_mask)
            if num_pruned > 1100:
                iteration = 1
                
        #num_iters = num_iterations * epoch + i

        if i > 100 and top1.val < 30:
            iteration = 1       
        #iteration = 2
        gl.set_value('iteration', iteration)

        inputs = torch.cat([data[j]["data"] for j in range(num_gpu)], dim=0)
        targets = torch.cat([data[j]["label"] for j in range(num_gpu)], dim=0).squeeze().long()

        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()  

        #inputs = inputs.to(args.gpus[0])
        #targets = targets.to(args.gpus[0])
        features_t = model_t(inputs)
        features_s = model_s(inputs)
        #features_kd = model_kd(inputs)

        ############################
        # (1) Update
        # D network
        ###########################
        #''' 
        for p in model_d.parameters():
            p.requires_grad = True

        optimizer_d.zero_grad()

        output_t = model_d(features_t.to(args.gpus[0]).detach())

        labels_real = torch.full_like(output_t, real_label, device=args.gpus[0])
        error_real = bce_logits(output_t, labels_real)

        output_s = model_d(features_s.to(args.gpus[0]).detach())

        labels_fake = torch.full_like(output_t, fake_label, device=args.gpus[0])
        error_fake = bce_logits(output_s, labels_fake)

        error_d = 0.1 * error_real + 0.1 * error_fake

        labels = torch.full_like(output_s, real_label, device=args.gpus[0])

        #error_d += bce_logits(output_s, labels)
        error_d.backward()

        #losses_d.update(error_d.item(), inputs.size(0))
        #writer_train.add_scalar(
            #'discriminator_loss', error_d.item(), num_iters)

        optimizer_d.step()
        #if i % args.print_freq == 0:#i >= 0:#
        if i < 0:
            print(
                '=> D_Epoch[{0}]({1}/{2}):\t'
                'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
                    epoch, i, num_iterations, loss_d=losses_d))

        #'''        
        ############################
        # (2) Update student network
        ###########################

        #'''

        for p in model_d.parameters():
            p.requires_grad = False

        optimizer_s.zero_grad()
        optimizer_m.zero_grad()

        alpha = 0.9 - epoch / args.num_epochs * 0.9
        Temperature = 10
        KD_loss = 10 * nn.KLDivLoss()(F.log_softmax(features_s / Temperature, dim=1),
                                 F.softmax(features_t / Temperature, dim=1)) * (
                              alpha * Temperature * Temperature) + F.cross_entropy(features_s, targets) * (1 - alpha)
        KD_loss.backward(retain_graph=True)
        #losses_kl.update(KD_loss.item(), inputs.size(0))





        # data_loss
        alpha = 0.9 - epoch / args.num_epochs * 0.9
        #one_hot = torch.zeros(targets.shape[0], 1000).cuda()
        #one_hot = one_hot.scatter_(1, targets.reshape(targets.shape[0],1), 1).cuda()
        error_data = args.miu * (alpha * F.mse_loss(features_t, features_s.to(args.gpus[0])))# + (1 - alpha) * F.mse_loss(one_hot, features_s.to(args.gpus[0])))
        #losses_data.update(error_data.item(), inputs.size(0))
        error_data.backward(retain_graph=True)

        # fool discriminator
        #tt3 = time.time()
        output_s = model_d(features_s.to(args.gpus[0]))
        labels = torch.full_like(output_s, real_label, device=args.gpus[0])
        error_g = 0.1 * bce_logits(output_s, labels)
        #losses_g.update(error_g.item(), inputs.size(0))
        #writer_train.add_scalar(
            #'generator_loss', error_g.item(), num_iters)
        error_g.backward(retain_graph=True)

        optimizer_s.step()
         
        #'''

        # train mask
        error_sparse = 0
        decay = (epoch % args.lr_decay_step == 0 and i == 1)
        if i % (args.mask_step) == 0:
            mask = []
            for name, param in model_s.named_parameters():
                if 'mask' in name:
                    mask.append(param.view(-1))
            mask = torch.cat(mask)
            error_sparse = 0.00001 * args.sparse_lambda * F.l1_loss(mask, torch.zeros(mask.size()).to(args.gpus[0]), reduction='sum')
            error_sparse.backward()
            optimizer_m.step(decay)
            #losses_sparse.update(error_sparse.item(), inputs.size(0))
            #writer_train.add_scalar(
            #'sparse_loss', error_sparse.item(), num_iters)
        prec1, prec5 = utils.accuracy(features_s.to(args.gpus[0]), targets.to(args.gpus[0]), topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))


        if i % 60 == 0:
            t1 = time.time()
            print(
                '=> G_Epoch[{0}]({1}/{2}):\n'
                'Loss_s {loss_sparse:.4f} \t'
                'Loss_data {loss_data:.4f}\t'
                'Loss_d {loss_d:.4f} \n'
                'Loss_g {loss_g:.4f} \t'
                'Loss_kl {loss_kl:.4f} \n'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                'time {time:.4f}\t'
                'pruned {np}'.format(
                    epoch, i, num_iterations, loss_sparse=error_sparse, loss_data=error_data, loss_d=error_d,
                    loss_g=error_g, loss_kl=KD_loss, top1=top1, top5=top5, time = t1-t0, np = num_pruned))
            logging.info(
                'TRAIN epoch: %03d step : %03d  Top1: %e Top5: %e error_g: %e error_data: %e error_d: %e Duration: %f Pruned: %d',
                epoch, i, top1.avg, top5.avg, error_g, error_data, error_d, t1 - t0, num_pruned)

        #'''
def test(args, loader_test, model_s):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()

    with torch.no_grad():
        #for i, (inputs, targets) in enumerate(loader_test, 1):
        for i, data in enumerate(loader_test):
            #if i > 20:
            #    break
            inputs = torch.cat([data[j]["data"] for j in range(num_gpu)], dim=0)
            targets = torch.cat([data[j]["label"] for j in range(num_gpu)], dim=0).squeeze().long()

            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda()
            #inputs = inputs.to(args.gpus[0])
            #targets = targets.to(args.gpus[0])

            logits = model_s(inputs).to(args.gpus[0])
            loss = cross_entropy(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print('* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info(
            'Top1: %e Top5: %e ',top1.avg, top5.avg)


        mask = []
        for name, weight in model_s.named_parameters():
            if 'mask' in name:
                for i in range(len(weight)):
                    mask.append(weight[i].item())

        # num_pruned = sum(m == 0 for m in mask)
        print("* Pruned {} / {}".format(sum(m == 0 for m in mask), len(mask)))
        logging.info('Pruned: %e  Total: %e ',sum(m == 0 for m in mask), len(mask))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
