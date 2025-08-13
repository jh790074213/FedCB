import os
import datetime
import logging
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

from options import args_parser

args = args_parser()


def log_init():
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    log_path = os.path.join(args.logdir, args.log_file_name + '.log')
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_path)])
    return logging.getLogger()


def tensorboard_writer(ten_dir):
    if not os.path.isdir(ten_dir):
        os.mkdir(ten_dir)
    if args.dataset == 'SVHN':
        if not os.path.isdir('{}/SVHN/{}'.format(ten_dir, args.time_current)):
            os.makedirs('{}/SVHN/{}'.format(ten_dir, args.time_current))
        writer = SummaryWriter('{}/SVHN/{}'.format(ten_dir, args.time_current))
    elif args.dataset == 'cifar100':
        if not os.path.isdir('{}/cifar100/{}'.format(ten_dir, args.time_current)):
            os.makedirs('{}/cifar100/{}'.format(ten_dir, args.time_current))
        writer = SummaryWriter('{}/cifar100/{}'.format(ten_dir, args.time_current))
    elif args.dataset == 'cifar10':
        if not os.path.isdir('{}/cifar10/{}'.format(ten_dir, args.time_current)):
            os.makedirs('{}/cifar10/{}'.format(ten_dir, args.time_current))
        writer = SummaryWriter('{}/cifar10/{}'.format(ten_dir, args.time_current))
    elif args.dataset == 'fmnist':
        if not os.path.isdir('{}/fmnist/{}'.format(ten_dir, args.time_current)):
            os.makedirs('{}/fmnist/{}'.format(ten_dir, args.time_current))
        writer = SummaryWriter('{}/fmnist/{}'.format(ten_dir, args.time_current))
    else:
        if not os.path.isdir('{}/skin/{}'.format(ten_dir, args.time_current)):
            os.makedirs('{}/skin/{}'.format(ten_dir, args.time_current))
        writer = SummaryWriter('{}/skin/{}'.format(ten_dir, args.time_current))
    return writer


def load_partition_strategy(partition_strategy_dir):
    assert os.path.isdir(partition_strategy_dir), 'Error: no partition_strategy directory found!'
    # 直接导入了对每个客户端进行数据分配.pth中只有个data_partition
    if args.dataset == 'SVHN':
        partition = torch.load('{}/SVHN_noniid_10%labeled.pth'.format(partition_strategy_dir))
        # 每个客户端拥有的数据索引{0：[..],1:[..]}
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        # partition = torch.load('{}/cifar100_noniid_beat_0.8.pth'.format(partition_strategy_dir))
        partition = torch.load('{}/cifar100_noniid_10%labeled.pth'.format(partition_strategy_dir))
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar10':
        partition = torch.load('{}/cifar10_noniid_beat_0.8.pth'.format(partition_strategy_dir))
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'fmnist':
        partition = torch.load('{}/fmnist_noniid_beat_0.8.pth'.format(partition_strategy_dir))
        net_dataidx_map = partition['data_partition']
    else:
        partition = torch.load('{}/skin_noniid_beta0.8.pth'.format(partition_strategy_dir))
        net_dataidx_map = partition['data_partition']
    return partition, net_dataidx_map
