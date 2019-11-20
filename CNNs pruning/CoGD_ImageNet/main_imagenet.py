import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import utils.common as utils
from utils.options import args
#from utils.preprocess import prune_resnet

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from fista import FISTA
from model import Discriminator, resnet_56, resnet_56_sparse
from data import cifar10

from resnet import ResNet18
from resnet_sprase import ResNet18_sprase

from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
from resnet import ResNet101

import utils.globalvar as gl
gl._init()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            # print (name)
            # print(module)
            if name is "linear":
                # print(x.size(), 'x.size@@@@')
                x = x.view(x.size(0), -1)
                # print(x.size(), 'x.size######')

            x = module(x)  # last layer output put into current layer input
            # print(x.size(), 'x.size')
            if name in self.extracted_layers:
                outputs.append(x)
        # print('!!!!!!!!!!!')
        return outputs


def main():
    checkpoint = utils.checkpoint(args)
    writer_train = SummaryWriter(args.job_dir + '/run/train')
    writer_test = SummaryWriter(args.job_dir + '/run/test')

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    loader = cifar10(args)

    # Create model
    print('=> Building model...')
    # model_t = resnet_56().to(args.gpus[0])
    #model_t = MobileNetV2()
    model_t = ResNet18()
    model_kd = ResNet101()
   
    print(model_kd)
    # Load teacher model
    ckpt_t = torch.load(args.teacher_dir, map_location=torch.device(f"cuda:{args.gpus[0]}"))
    state_dict_t = ckpt_t['net']
    new_state_dict_t = OrderedDict()

    new_state_dict_t = state_dict_t
    #for k, v in state_dict_t.items():
        #print(k[0:6])
        #if k[0:6] == 'linear':
            #temp = v[0:10]
            #print(v[0:10].shape)
            #new_state_dict_t[k] = temp

    #model_t.load_state_dict(new_state_dict_t)
    model_t = model_t.to(args.gpus[0])

    for para in list(model_t.parameters())[:-2]:
        para.requires_grad = False

    #model_s = SpraseMobileNetV2().to(args.gpus[0])
    model_s = ResNet18_sprase().to(args.gpus[0])
    print(model_s)
    model_dict_s = model_s.state_dict()
    model_dict_s.update(new_state_dict_t)
    model_s.load_state_dict(model_dict_s)


    ckpt_kd = torch.load('resnet101.t7', map_location=torch.device(f"cuda:{args.gpus[0]}"))
    state_dict_kd = ckpt_kd['net']
    new_state_dict_kd = OrderedDict()
    for k, v in state_dict_kd.items():
        name = k[7:]
        new_state_dict_kd[name] = v
    #print(new_state_dict_kd)
    model_kd.load_state_dict(new_state_dict_kd)
    model_kd = model_kd.to(args.gpus[0])

    for para in list(model_kd.parameters())[:-2]:
        para.requires_grad = False


    if len(args.gpus) != 1:
        print('@@@@@@')
        model_s = nn.DataParallel(model_s, device_ids=args.gpus[0,1])

    model_d = Discriminator().to(args.gpus[0])

    models = [model_t, model_s, model_d, model_kd]

    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_m = FISTA(param_m, lr=args.lr, gamma=args.sparse_lambda)

    scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=torch.device(f"cuda:{args.gpus[0]}"))
        best_prec1 = ckpt['best_prec1']
        model_s.load_state_dict(ckpt['state_dict_s'])
        model_d.load_state_dict(ckpt['state_dict_d'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        print('=> Continue from epoch {}...'.format(ckpt['epoch']))

    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]

    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return

    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)

        global g_e
        g_e = epoch
        gl.set_value('epoch',g_e)      


        #train(args, loader.loader_train, models, optimizers, epoch, writer_train)
        #print('###########################')
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)

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
        if is_best:
          checkpoint.save_model(state, epoch + 1, is_best)

    print(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

    # best_model = torch.load(f'{args.job_dir}/checkpoint/model_best.pt', map_location=torch.device(f"cuda:{args.gpus[0]}"))

    # model = prune_resnet(args, best_model['state_dict_s'])


def train(args, loader_train, models, optimizers, epoch, writer_train):
    losses_d = utils.AverageMeter()
    losses_data = utils.AverageMeter()
    losses_g = utils.AverageMeter()
    losses_sparse = utils.AverageMeter()
    losses_kl = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
    model_s = models[1]
    model_d = models[2]
    model_kd = models[3]

    bce_logits = nn.BCEWithLogitsLoss()

    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]
    optimizer_m = optimizers[2]

    # switch to train mode
    model_d.train()
    model_s.train()

    num_iterations = len(loader_train)

    real_label = 1
    fake_label = 0
    exact_list=["layer3"]

    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
        #print(i,'i')
        global iteration
        iteration = i
        gl.set_value('iteration', iteration)
        inputs = inputs.to(args.gpus[0])
        targets = targets.to(args.gpus[0])

        features_t = model_t(inputs)
        features_kd = model_kd(inputs)
        features_s = model_s(inputs)
        # FM_exactor_t = FeatureExtractor(model_t, exact_list).cuda()
        # FM_t = FM_exactor_t(inputs)
        #
        # FM_exactor_s = FeatureExtractor(model_s, exact_list).cuda()
        # FM_s = FM_exactor_s(inputs)
        # print(FM_t[0].shape,'@@@@@@@@@@@')
        # print(FM_s[0].shape, '@@@@@@@@@@@')
        ############################
        # (1) Update
        # D network
        ###########################

        for p in model_d.parameters():
            p.requires_grad = True

        optimizer_d.zero_grad()

        output_t = model_d(features_t.detach())

        # output_t = model_d(FM_t[0].detach())

        labels_real = torch.full_like(output_t, real_label, device=args.gpus[0])
        error_real = bce_logits(output_t, labels_real)

        output_s = model_d(features_s.to(args.gpus[0]).detach())
        # output_s = model_d(FM_s[0].detach())

        labels_fake = torch.full_like(output_t, fake_label, device=args.gpus[0])
        error_fake = bce_logits(output_s, labels_fake)

        error_d = error_real + error_fake

        labels = torch.full_like(output_s, real_label, device=args.gpus[0])

        error_d += bce_logits(output_s, labels)
        error_d.backward()
        losses_d.update(error_d.item(), inputs.size(0))
        writer_train.add_scalar(
            'discriminator_loss', error_d.item(), num_iters)

        optimizer_d.step()

        if i % args.print_freq == 0:
            print(
                '=> D_Epoch[{0}]({1}/{2}):\t'
                'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
                    epoch, i, num_iterations, loss_d=losses_d))

        ############################
        # (2) Update student network
        ###########################

        for p in model_d.parameters():
            p.requires_grad = False

        optimizer_s.zero_grad()
        optimizer_m.zero_grad()


        # reg = 1e-6
        # l1_loss = Variable(torch.FloatTensor(1), requires_grad=True).to(args.gpus[0])
        # for name, param in model_s.named_parameters():
        #     if 'bias' not in name:
        #         l1_loss = l1_loss+(reg * torch.sum(torch.abs(param)))

        # print(l1_loss, '!!!!!!')


        # print(features_t.shape,'features_t')
        # print(features_t.shape,'features_s')
        # print(targets.shape,'targets')

        # error_data_kl = nn.functional.kl_div(features_s, features_t.float())
        # KL_loss = nn.KLDivLoss(reduction='sum')
        # log_fea_s = F.log_softmax(features_s, 1)
        # fea_t = F.softmax(features_t, 1)
        # # fea_t = nn.functional.log_softmax(features_t)
        # # fea_s = nn.functional.log_softmax(features_s)
        # error_data_kl = KL_loss(log_fea_s, fea_t.float())/inputs.size(0)
        # loss_fun = nn.CrossEntropyLoss()
        # error_data_kl = loss_fun(features_s, features_t)
        # error_data_kl.backward(retain_graph=True)
        alpha = 0.99
        Temperature = 30
        KD_loss = nn.KLDivLoss()(F.log_softmax(features_s / Temperature, dim=1),
                                 F.softmax(features_kd / Temperature, dim=1)) * (
                              alpha * Temperature * Temperature) + F.cross_entropy(features_s, targets) * (1 - alpha)
        KD_loss.backward(retain_graph=True)
        # print(error_data_kl,'error_data_kl')
        # print(error_data,'error_data')
        # error_data += error_data_kl
        losses_kl.update(KD_loss.item(), inputs.size(0))

        error_data = args.miu * F.mse_loss(features_t, features_s.to(args.gpus[0]))  ##
        losses_data.update(error_data.item(), inputs.size(0))
        writer_train.add_scalar(
            'data_loss', error_data.item(), num_iters)
        error_data.backward(retain_graph=True)

        # fool discriminator
        output_s = model_d(features_s.to(args.gpus[0]))
        # output_s = model_d(FM_s[0])

        labels = torch.full_like(output_s, real_label, device=args.gpus[0])
        error_g = bce_logits(output_s, labels)
        losses_g.update(error_g.item(), inputs.size(0))
        writer_train.add_scalar(
            'generator_loss', error_g.item(), num_iters)
        error_g.backward(retain_graph=True)

        # train mask
        mask = []
        for name, param in model_s.named_parameters():
            if 'mask' in name:
                mask.append(param.view(-1))
        mask = torch.cat(mask)
        error_sparse = 0.01 * args.sparse_lambda * F.l1_loss(mask, torch.zeros(mask.size()).to(args.gpus[0]),
                                                             reduction='sum')
        error_sparse.backward()

        losses_sparse.update(error_sparse.item(), inputs.size(0))
        writer_train.add_scalar(
            'sparse_loss', error_sparse.item(), num_iters)

        optimizer_s.step()

        decay = (epoch % args.lr_decay_step == 0 and i == 1)
        if i % args.mask_step == 0:
            optimizer_m.step(decay)

        prec1, prec5 = utils.accuracy(features_s.to(args.gpus[0]), targets.to(args.gpus[0]), topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if i % args.print_freq == 0:
            print(
                '=> G_Epoch[{0}]({1}/{2}):\t'
                'Loss_sparse {loss_sparse.val:.4f} ({loss_sparse.avg:.4f})\t'
                'Loss_data {loss_data.val:.4f} ({loss_data.avg:.4f})\t'
                'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                'Loss_g {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                'Loss_kl {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, num_iterations, loss_sparse=losses_sparse, loss_data=losses_data, loss_g=losses_g,
                    loss_d=losses_d, loss_kl=losses_kl, top1=top1, top5=top5))


def test(args, loader_test, model_s):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            inputs = inputs.to(args.gpus[0])
            targets = targets.to(args.gpus[0])

            logits = model_s(inputs).to(args.gpus[0])
            loss = cross_entropy(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print('* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        mask = []
        for name, weight in model_s.named_parameters():
            if 'mask' in name:
                for i in range(len(weight)):
                    mask.append(weight[i].item())

        print("* Pruned {} / {}".format(sum(m == 0 for m in mask), len(mask)))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
