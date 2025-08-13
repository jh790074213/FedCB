import numpy as np
from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import ModelFedCon
from utils import ramps
from utils import loss_tool
import torch.nn as nn
from utils.utils_SimPLE import label_guessing, sharpen
# from loss.loss import UnsupervisedLoss  # , build_pair_loss
import logging
from torchvision import transforms
from ramp import LinearRampUp

args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# 教师参数更新，EMA方式接收学生参数 alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    # alpha逐渐增大
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, weak_aug, strong_aug, label = self.dataset[self.idxs[item]]
        return items, index, weak_aug, strong_aug, label


class UnsupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
        net_ema = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
            net_ema = torch.nn.DataParallel(net_ema, device_ids=[i for i in range(round(len(args.gpu) / 2))])
        # 教师模型
        self.ema_model = net_ema.cuda()

        # 学生模型
        self.model = net.cuda()

        # 确保 EMA 模型的参数在训练过程中不参与梯度计算。
        for param in self.ema_model.parameters():
            param.detach_()

        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        # self.mse_loss = nn.MSELoss()
        # 是否是第一次训练，是则教师模型参数用全局模型参数初始化
        self.flag = True
        self.unsup_lr = args.unsup_lr
        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.unsup_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9,
                                             weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.unsup_lr,
                                               weight_decay=0.02)

    def train(self, args, net_w, op_dict, epoch, unlabeled_idx, train_dl_local, n_classes, local_confi, class_weight,
              num_client):
        # 学生模型初始化
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.ema_model.load_state_dict(copy.deepcopy(net_w))
        self.model.train()
        self.ema_model.eval()

        self.model.cuda()
        self.ema_model.cuda()
        # 初始化优化器状态字典
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.unsup_lr

        # self.epoch = epoch
        # 教师模型第一次训练初始化
        if self.flag:
            self.ema_model.load_state_dict(copy.deepcopy(net_w))
            self.flag = False
            logging.info('EMA model initialized')
        # 教师模型参数跳跃连接
        # if epoch % 5 == 0:
        # if epoch == 2:
        #     print("hello")
        # update_ema_variables(self.model, self.ema_model, args.ema_decay, epoch)

        epoch_loss = []
        logging.info('Unlabeled client %d begin unsupervised training' % unlabeled_idx)

        # 记录输出的每个类别的数量
        epoch_num_class = []
        epoch_num_class_l = []
        pred_cli_max = []
        pl_num = 0
        for epoch in range(args.local_ep):
            batch_loss = []
            iter_num_class = [0 for _ in range(n_classes)]
            iter_num_class_l = [0 for _ in range(n_classes)]
            pl_loss_epoch = []
            loss_u_tea_stu_epoch = []
            # 保存每个类的预测值
            pred_cli_epoch = np.zeros(n_classes)
            nums_cli_epoch = np.zeros(n_classes)

            for i, (_, weak_aug_batch, label_batch) in enumerate(train_dl_local):
                weak_aug_batch = [weak_aug_batch[version].cuda() for version in range(len(weak_aug_batch))]
                # 不计算梯度下执行
                with torch.no_grad():
                    # if unlabeled_idx == 4 and i == 81:
                    #     print(4)
                    # weak_aug_batch包含增强后的两个数据[0]输入教师，label_guessing获得一个batch的预测值
                    guessed = label_guessing(self.ema_model, batches=weak_aug_batch[0], model_type=args.model)
                    # sharpening
                    sharpened = sharpen(guessed)
                # 教师输出标签
                pseu = torch.argmax(guessed, dim=1)
                label = label_batch.squeeze()
                if len(label.shape) == 0:
                    label = label.unsqueeze(dim=0)
                label = label.cuda()

                # 增强[1]输入学生模型
                logits_str = self.model(weak_aug_batch[1], model=args.model)[2]
                # 学生模型输出的预测值
                probs_str = F.softmax(logits_str, dim=1)
                # TODO
                # logits_str_label = self.model(weak_aug_batch[0], model=args.model)[2]
                # probs_str_label = F.softmax(logits_str_label, dim=1)
                # probs_str_label = torch.argmax(probs_str_label, dim=1)

                # max_pred最大预测值，pred_classes对应的类别索引
                # max_pred, pred_classes = torch.max(probs_str, dim=1)

                # 教师模型输出标签
                max_pred_tea, pred_classes_tea = torch.max(guessed, dim=1)
                # 预测值大于阈值的样本索引
                mask = max_pred_tea > torch.tensor(local_confi).cuda()[pred_classes_tea]
                pl_num += mask.sum().item()
                # 统计伪标签预测值
                for pred, classes in zip(max_pred_tea[mask], pred_classes_tea[mask]):
                    pred_cli_epoch[classes] += pred
                    nums_cli_epoch[classes] += 1
                # 预测值最大值
                pred_cli_max.append((torch.sum(max_pred_tea) / len(max_pred_tea)).tolist())
                # 统计教师模型的类别数量
                for pred_label in pseu[mask]:
                    iter_num_class[pred_label] += 1
                # 统计真实标签
                for true_label in label[mask]:
                    iter_num_class_l[int(true_label.item())] += 1

                if len(guessed[mask]) > 0:
                    # pl_loss_fn = torch.nn.CrossEntropyLoss()
                    pl_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight[num_client], dtype=torch.float32).cuda())
                    pl_loss = pl_loss_fn(probs_str[mask], guessed[mask])

                    # # 学生模型输出标签
                    # pred_labels = torch.argmax(probs_str, dim=1)
                    # # 统计预测的每个类别数据量
                    # for pred_label in pred_labels:
                    #     iter_num_class[pred_label] += 1

                    # # 将预测转为one-hot 作为为标签
                    # pl_label = torch.zeros(len(pred_labels), n_classes, requires_grad=True).cuda()
                    # pl_label.scatter_(1, pred_labels.view(-1, 1), 1)

                    # 伪标签损失
                    # pl_loss_fn = torch.nn.CrossEntropyLoss()
                    # pl_loss = pl_loss_fn(pl_label[mask], probs_str[mask])
                    # pl_num += len(pl_label[mask])

                    # 无监督学习损失,mse会除元素向量长度100，再除样本数64，原代码直接除样本数量
                    # loss_u_tea_stu = torch.sum(loss_tool.softmax_mse_loss(probs_str[mask], sharpened[mask])) / len(
                    #     probs_str[mask])
                    # loss_u_tea_stu = torch.sum(loss_tool.softmax_mse_loss(pl_label[mask], sharpened[mask])) / len(
                    #     probs_str[mask])
                    # loss = args.lambda_pl * pl_loss + args.lambda_u * loss_u_tea_stu

                    # loss = args.lambda_u * loss_u_tea_stu
                    loss = pl_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                    self.optimizer.step()
                    # 教师参数更新
                    # update_ema_variables(self.model, self.ema_model, args.ema_decay, self.iter_num)
                    batch_loss.append(loss.item())
                    pl_loss_epoch.append(pl_loss.item())
                    # loss_u_tea_stu_epoch.append(loss_u_tea_stu.item())
                self.iter_num = self.iter_num + 1

            # 计算这轮通信的每个类别的平均预测值
            with np.errstate(divide='ignore', invalid='ignore'):
                pred_cli_epoch = np.divide(pred_cli_epoch, nums_cli_epoch)
                pred_cli_epoch[nums_cli_epoch == 0] = 0
            if len(pl_loss_epoch) > 0:
                print('pl_loss=========>{}'.format(sum(pl_loss_epoch) / len(pl_loss_epoch)))
                # print('loss_u_tea_stu=========>{}'.format(sum(loss_u_tea_stu_epoch) / len(loss_u_tea_stu_epoch)))
            print('大于阈值数量:{}'.format(pl_num))
            print('平均预测值pred_cli_epoch:{}'.format(pred_cli_epoch))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_num_class.append(np.array(iter_num_class))
            epoch_num_class_l.append(np.array(iter_num_class_l))
            print("=====================未标记客户端{}实际类别数量==================".format(unlabeled_idx))
            print(np.round(np.mean(epoch_num_class_l, axis=0)))
            self.epoch = self.epoch + 1


        self.model.cpu()
        self.ema_model.cpu()
        return (self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict()), pred_cli_epoch, np.round(np.mean(epoch_num_class, axis=0)),
                np.mean(np.array(pred_cli_max)), pl_num)

        # return self.model.state_dict(), self.ema_model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
        #     self.optimizer.state_dict()), local_pred_exp_cli, np.round(
        #     np.mean(epoch_num_class, axis=0)), np.mean(np.array(pred_cli_max))
