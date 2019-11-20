import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import math

try:
    import nvidia.dali.plugin.pytorch as plugin_pytorch
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

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
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
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


# def getTrainValDataset(traindir, valdir, batch_size, val_batch_size, num_shards, workers):
#     pipes = [
#         HybridTrainPipe(batch_size=int(batch_size / num_shards), num_threads=workers,
#                         device_id=device_id,
#                         data_dir=traindir, crop=224, num_shards=num_shards) for device_id in
#         range(num_shards)]
#     pipes[0].build()
#     train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))
#
#     pipes = [
#         HybridTrainPipe(batch_size=int(val_batch_size / num_shards), num_threads=workers,
#                         device_id=device_id,
#                         data_dir=valdir, crop=224, num_shards=num_shards) for device_id in
#         range(num_shards)]
#     pipes[0].build()
#     val_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))
#     return train_loader, val_loader
def getTrainValDataset(traindir, valdir, batch_size, val_batch_size, num_shards, workers):
    pipes = [
        HybridTrainPipe(batch_size=int(batch_size / num_shards), num_threads=workers,
                        device_id=device_id,
                        data_dir=traindir, crop=224, num_shards=num_shards) for device_id in
        range(num_shards)]
    pipes[0].build()
    train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))
    # pipes = [
    #     HybridTrainPipe(batch_size=int(val_batch_size / num_shards), num_threads=workers,
    #                     device_id=device_id,
    #                     data_dir=valdir, crop=224, num_shards=num_shards) for device_id in
    #     range(num_shards)]
    # pipes[0].build()
    # val_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

    return train_loader#, val_loader


def getTestDataset(testdir, test_batch_size, num_shards, workers):
    pipes = [
        HybridValPipe(batch_size=int(test_batch_size / num_shards), num_threads=workers, device_id=device_id,
                      data_dir=testdir,
                      crop=224, size=256, num_shards=num_shards) for device_id in range(num_shards)]
    pipes[0].build()
    test_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

    return test_loader


def get_loaders(train_portion, batch_size, path_to_save_data, logger):
    traindir = os.path.join(path_to_save_data, 'ILSVRC2012_img_train_rec')
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]))

    num_train = len(train_data)  # 50k
    indices = list(range(num_train))  #
    split = int(np.floor(train_portion * num_train))  # 40k

    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=32)

    if train_portion == 1:
        return train_loader

    valid_sampler = SubsetRandomSampler(valid_idx)

    val_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler,
        pin_memory=True, num_workers=16)

    return train_loader, val_loader


def get_test_loader(batch_size, path_to_save_data):
    testdir = os.path.join(path_to_save_data, 'ILSVRC2012_img_val_rec')
    test_data = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]))

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return test_loader
#
# import numpy as np
# import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# import os
#
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
#
#
# def get_loaders(train_portion, batch_size, path_to_save_data, logger):
#     traindir = os.path.join(path_to_save_data, 'ILSVRC2012_img_train100')
#     train_data = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomSizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#         ]))
#
#     num_train = len(train_data)  # 50k
#     indices = list(range(num_train))  #
#     split = int(np.floor(train_portion * num_train))  # 40k
#
#     train_idx, valid_idx = indices[:split], indices[split:]
#
#     train_sampler = SubsetRandomSampler(train_idx)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=batch_size, sampler=train_sampler,
#         pin_memory=True, num_workers=32)
#
#     if train_portion == 1:
#         return train_loader
#
#     valid_sampler = SubsetRandomSampler(valid_idx)
#
#     val_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=batch_size, sampler=valid_sampler,
#         pin_memory=True, num_workers=32)
#
#     return train_loader, val_loader
#
#
# def get_test_loader(batch_size, path_to_save_data):
#     testdir = os.path.join(path_to_save_data, 'ILSVRC2012_img_val100')
#     test_data = datasets.ImageFolder(testdir, transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ]))
#
#     test_loader = torch.utils.data.DataLoader(
#         test_data, batch_size=batch_size, shuffle=False,
#         num_workers=32, pin_memory=True)
#
#     return test_loader