import numpy as np
import torch
import torch.optim
from options import args_parser
import copy
# from utils import losses
import logging
# from pytorch_metric_learning import losses
from networks.models import ModelFedCon
import torch.nn.functional as F

args = args_parser()


class SupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        # 本地训练轮数gi
        self.epoch = 0
        # 一轮训练迭代次数
        self.iter_num = 0

        self.base_lr = args.base_lr
        self.data_idx = idxs
        #  梯度的最大值，防止梯度爆炸
        self.max_grad_norm = args.max_grad_norm

        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
        # 模型初始化
        self.model = net.cuda()

    def train(self, args, net_w, op_dict, dataloader, n_classes,num_client, class_weight):
        # 加载全局模型
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.cuda().train()
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args.base_lr, momentum=0.9,
                                             weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.base_lr,
                                               weight_decay=0.02)
            # SimPLE original paper: lr=0.002, weight_decay=0.02
        #  初始化优化器状态字典
        self.optimizer.load_state_dict(op_dict)

        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        # 载入权重
        # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight[num_client], dtype=torch.float32).cuda())

        loss_fn = torch.nn.CrossEntropyLoss()

        # 记录输出的每个类别的数量
        epoch_num_class = []
        epoch_num_class_pd = []
        epoch_loss = []
        logging.info('========Begin supervised training=========')

        # 预测值最大值
        pred_cli_max = []

        # 保存每个类的预测值
        pred_cli_epoch = np.zeros(n_classes)
        nums_cli_epoch = np.zeros(n_classes)
        for epoch in range(args.sup_local_ep):
            batch_loss = []
            iter_num_class = [0 for _ in range(n_classes)]
            # 预测类别数
            iter_num_class_pd = [0 for _ in range(n_classes)]
            for i, (_, image_batch, label_batch) in enumerate(dataloader):
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                label_batch = label_batch.long().squeeze()
                inputs = image_batch
                # 模型的输出和激活值。
                _, activations, outputs = self.model(inputs, model=args.model)
                suo_pred = F.softmax(outputs, dim=1)

                # 确保在后续的计算中（例如计算交叉熵损失时），label_batch 和 outputs 具有正确的形状。
                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)
                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)

                # 统计最大预测值
                max_pred, pred_classes = torch.max(suo_pred, dim=1)

                pred_cli_max.append((torch.sum(max_pred) / len(max_pred)).tolist())
                # 统计预测值
                for pred, classes in zip(max_pred, pred_classes):
                    pred_cli_epoch[classes] += pred
                    nums_cli_epoch[classes] += 1

                # 统计预测的每个类别数据量
                pred_labels = torch.argmax(outputs, dim=1)
                for pred_label in pred_labels:
                    iter_num_class_pd[pred_label] += 1
                # 真实标签
                for pred_label in label_batch:
                    iter_num_class[pred_label] += 1

                loss = loss_fn(outputs, label_batch)
                self.optimizer.zero_grad()
                loss.backward()
                # 对模型参数的梯度进行梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()
                batch_loss.append(loss.item())
                self.iter_num = self.iter_num + 1

            # 计算这轮通信的每个类别的平均预测值
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_cli_epoch = np.divide(pred_cli_epoch, nums_cli_epoch)
                pred_cli_epoch[nums_cli_epoch == 0] = 0

            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            epoch_num_class.append(np.array(iter_num_class))
            epoch_num_class_pd.append(np.array(iter_num_class_pd))
            print("=====================标记客户端预测类别数量==================")
            print(np.round(np.mean(epoch_num_class_pd, axis=0)))

        self.model.cpu()

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict()), np.round(np.mean(epoch_num_class, axis=0)), np.mean(
            np.array(pred_cli_max)), pred_cli_epoch
