from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
from FedAvg import FedAvg, model_dist
import torch
import torch.backends.cudnn as cudnn
from networks.models import ModelFedCon
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from tqdm import trange
from dataloader.cifar_load import get_dataloader, partition_data_allnoniid
from utils.utils import log_init, tensorboard_writer, load_partition_strategy


def test(epoch, checkpoint, data_test, label_test, n_classes):
    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'fmnist':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus, Pre, Recall = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg


def global_confi_update(global_confi, alpha, com_round, freq_global_confi):
    alpha = min(1 - 1 / (com_round + 1), alpha)
    return alpha * global_confi + (1 - alpha) * freq_global_confi


# 更新预测期望
def local_pred_exp_cli_update(local_pred_exp_cli, local_pred_exp_next, alpha, global_step):
    next_pred_exp_cli = []
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for pred_exp, pred_mean in zip(local_pred_exp_cli, local_pred_exp_next):
        pred_exp = pred_exp * alpha + pred_mean * (1 - alpha)
        next_pred_exp_cli.append(pred_exp.item())
    return next_pred_exp_cli


if __name__ == '__main__':
    args = args_parser()
    # 客户端比例
    supervised_user_id = list(range(0, args.sup_num))
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    if args.warmup:
        unsupervised_user_id = []
    total_num = sup_num + unsup_num
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    time_current = args.time_current
    # 日志
    logger = log_init()
    # 添加控制台输出，log_init()初始化中默认只有文件输出
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(str(args))
    logger.info(time_current)
    # 确定性训练，保证结果一致
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    # tensorboard
    writer = tensorboard_writer('tensorboard')

    # 创建目录保存模型参数
    snapshot_path = 'model/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == 'SVHN':
        snapshot_path = 'model/SVHN/'
    if args.dataset == 'cifar100':
        snapshot_path = 'model/cifar100/'
    if args.dataset == 'cifar10':
        snapshot_path = 'model/cifar10/'
    if args.dataset == 'fmnist':
        snapshot_path = 'model/fmnist/'
    if args.dataset == 'skin':
        snapshot_path = 'model/skin/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)

    if not os.path.isdir(snapshot_path + time_current):
        os.mkdir(snapshot_path + time_current)
    save_data_path = os.path.join('./',
                                  args.dataset + '_' + args.partition + '_beat_' + str(args.beta) + '.pth')

    print('================== Reloading data partitioning strategy..======================')
    if args.dataset == 'cifar10' or args.dataset == 'SVHN' or args.dataset == 'fmnist':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'skin':
        n_classes = 7
    # 预设分区策略
    # partition, net_dataidx_map = load_partition_strategy('partition_strategy')
    # 手动划分数据集 args.beta：dirichlet parameter 0.8 ；partition：noniid 并没有起到作用，只是加载数据
    if args.dataset != 'skin':
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data_allnoniid(
            args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta,
            n_classes=n_classes,
            unsup_num=unsup_num)
    else:
        # skin采用预设分区
        partition, net_dataidx_map = load_partition_strategy('partition_strategy')
        checkpoint = torch.load('partition_strategy/skin_noniid_beta0.8.pth')
        train_idxs = checkpoint['train_list']
        test_idxs = checkpoint['test_list']
        X_train, y_train, X_test, y_test = partition_data_allnoniid(args.dataset, args.datadir,
                                                                    partition=args.partition, n_parties=total_num,
                                                                    beta=args.beta,
                                                                    n_classes=n_classes, unsup_num=unsup_num,
                                                                    train_idxs=train_idxs, test_idxs=test_idxs)
    # torch.save({
    #     'data_partition': net_dataidx_map,
    # }
    #     , save_data_path
    # )

    # 将数据集中的图像数据从 (N, C, H, W) 形状转换为 (N, H, W, C) 形状
    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    # 加载全局模型
    net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    # 使用warmup 初始化全局模型参数和优化器参数 使用预热则从第几轮通信开始训练
    if args.resume:
        print('==> Resuming from checkpoint..')
        if args.dataset == 'cifar100':
            if args.model == 'simple-cnn':
                checkpoint = torch.load('warmup/cifar100.pth')
            else:
                checkpoint = torch.load('warmup/cifar100_resnet18.pth')
        elif args.dataset == 'cifar10':
            if args.model == 'simple-cnn':
                checkpoint = torch.load('warmup/cifar10.pth')
            else:
                checkpoint = torch.load('warmup/cifar10_resnet18.pth')
        elif args.dataset == 'SVHN':
            checkpoint = torch.load('warmup/SVHN.pth')
        elif args.dataset == 'skin':
            checkpoint = torch.load('warmup/skin_warmup.pth')
        elif args.dataset == 'fmnist':
            checkpoint = torch.load('warmup/fmnist_warmup.pth')
        net_glob.load_state_dict(checkpoint['state_dict'])
        global_confi = checkpoint['global_confi']
        local_confi = checkpoint['local_confi']
        local_pred_exp = checkpoint['local_pred_exp']
        # start_epoch = checkpoint['start_epoch']
    else:
        # 全局阈值
        global_confi = 1 / n_classes
        # 本地阈值
        local_confi = [1 / n_classes for i in range(n_classes)]
        # 初始化每个客户端每个类别预测期望
        local_pred_exp = [1 / n_classes for i in range(n_classes)]
    # 这里直接从第0轮开始
    start_epoch = 0
    # 保存类别权重
    class_weight = np.ones((total_num, n_classes))
    # 多gpu计算
    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])

    net_glob.train()
    w_glob = net_glob.state_dict()
    # 本地模型参数
    w_locals = []
    # 教师模型参数
    # w_ema_unsup = []
    # 有标签训练器
    lab_trainer_locals = []
    # 无标签训练器
    unlab_trainer_locals = []
    # 有标签神经网络
    sup_net_locals = []
    # 无标签神经网络
    unsup_net_locals = []
    # 有标签优化器参数
    sup_optim_locals = []
    # 无标签优化器参数
    unsup_optim_locals = []

    # 总的数据量
    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    # 每个客户端数据量
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    print(each_lenth)

    # 初始化监督训练
    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        sup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
        # 使用预热训练
        if args.resume:
            optimizer.load_state_dict(checkpoint['sup_optimizers'][i])
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
    # 初始化无监督训练 1-9 sup_num = 1
    for i in unsupervised_user_id:
        unlab_trainer_locals.append(
            UnsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        # w_ema_unsup.append(copy.deepcopy(w_glob))
        unsup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(unsup_net_locals[i - sup_num].parameters(),
                                        lr=args.unsup_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                          weight_decay=0.02)

        # 学生模型优化器参数初始化
        if args.resume and len(checkpoint['unsup_optimizers']) != 0:
            optimizer.load_state_dict(checkpoint['unsup_optimizers'][i - sup_num])
        unsup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

        # 预热模型是否只在有标签下得到，如果是则不进行下面操作
        # if args.resume and len(checkpoint['unsup_ema_state_dict']) != 0 and not args.from_labeled:
        #     w_ema_unsup = copy.deepcopy(checkpoint['unsup_ema_state_dict'])
        #     unlab_trainer_locals[i - sup_num].ema_model.load_state_dict(w_ema_unsup[i - sup_num])
        #     unlab_trainer_locals[i - sup_num].flag = False
        #     print('Unsup EMA reloaded')

    # 全局通信
    for com_round in trange(start_epoch, args.rounds + 1):
        logging.info("====================== Communication round %d begins ==========================" % com_round)
        # 保存这一轮通信的损失
        loss_locals = []
        # 该通信轮选择的客户端数组
        clt_this_comm_round = supervised_user_id + unsupervised_user_id
        # 保存该通信轮子共识模型
        w_com_round_local = []
        # 每个客户端数据量
        each_lenth_com_round = []
        for i in supervised_user_id:
            each_lenth_com_round.append(each_lenth[i])
        # 每个客户端预测的每个类别的数量
        num_class = []
        # 每个客户端输出最大概率的平均值
        pred_clis = []
        # 保存预测期望的更新
        local_pred_exp_next = []

        for i in range(len(local_confi)):
            if local_confi[i] > 0.95:
                local_confi[i] = 0.95
            # if local_confi[i] < 0.84:
            #     local_confi[i] = 0.84
        # TODO
        # for i in range(len(local_confi)):
        #     local_confi[i] = 0

        print('====================本地阈值====================')
        print(local_confi)

        # 本地训练
        for client_idx in clt_this_comm_round:
            # 监督训练
            if client_idx in supervised_user_id:
                local = lab_trainer_locals[client_idx]
                optimizer = sup_optim_locals[client_idx]
                # 得到客户端数据集的类加载器train_dl_local和预处理后数据集train_ds_local
                train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                y_train[net_dataidx_map[client_idx]],
                                                                args.dataset, args.datadir, args.batch_size,
                                                                is_labeled=True,
                                                                data_idxs=net_dataidx_map[client_idx],
                                                                pre_sz=args.pre_sz, input_sz=args.input_sz)
                # 本地训练 network, loss, optimizer
                w, loss, op, label_cli_num_class, pred_cli, local_pred_exp_cli = local.train(args, sup_net_locals[
                    client_idx].state_dict(), optimizer, train_dl_local, n_classes, num_client=client_idx,
                                                                                             class_weight=class_weight)

                writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)
                num_class.append(label_cli_num_class)
                w_com_round_local.append(copy.deepcopy(w))
                pred_clis.append(copy.deepcopy(pred_cli))
                local_pred_exp_next.append(copy.deepcopy(local_pred_exp_cli))
                sup_optim_locals[client_idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))
                logger.info(
                    'Labeled client {} sample num: {} training loss : {} lr : {}'.format(client_idx,
                                                                                         len(train_ds_local),
                                                                                         loss,
                                                                                         sup_optim_locals[
                                                                                             client_idx][
                                                                                             'param_groups'][0][
                                                                                             'lr']))
            # 完全未标记客户端训练
            else:
                local = unlab_trainer_locals[client_idx - sup_num]
                optimizer = unsup_optim_locals[client_idx - sup_num]
                train_dl_local, train_ds_local = get_dataloader(args,
                                                                X_train[net_dataidx_map[client_idx]],
                                                                y_train[net_dataidx_map[client_idx]],
                                                                args.dataset,
                                                                args.datadir, args.batch_size, is_labeled=False,
                                                                data_idxs=net_dataidx_map[client_idx],
                                                                pre_sz=args.pre_sz, input_sz=args.input_sz)
                w, loss, op, local_pred_exp_cli, unlabel_cli_num_class, pred_cli, pl_num = local.train(
                    args,
                    unsup_net_locals[client_idx - sup_num].state_dict(),
                    optimizer,
                    com_round * args.local_ep,
                    client_idx,
                    train_dl_local, n_classes,
                    local_confi=local_confi,
                    class_weight=class_weight,
                    num_client=client_idx)
                writer.add_scalar('Unsupervised loss on unsup client %d' % client_idx, loss, global_step=com_round)
                each_lenth_com_round.append(pl_num)
                w_com_round_local.append(copy.deepcopy(w))
                pred_clis.append(copy.deepcopy(pred_cli))
                # w_ema_unsup[client_idx - sup_num] = copy.deepcopy(w_ema)
                unsup_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))
                num_class.append(unlabel_cli_num_class)
                local_pred_exp_next.append(copy.deepcopy(local_pred_exp_cli))
                logger.info(
                    'Unlabeled client {} sample num: {} Training loss: {}, lr {},total {}'.format(
                        client_idx, len(train_ds_local), loss,
                        unsup_optim_locals[
                            client_idx - sup_num][
                            'param_groups'][
                            0]['lr'],
                        len(net_dataidx_map[client_idx]))
                )

        # # 增加有监督模型的聚合权重：将该客户端样本数量 * w_mul_times
        # if args.w_mul_times != 1 and 0 in clt_this_comm_round and (args.un_dist == '' or args.un_dist_onlyunsup):
        #     for sup_idx in supervised_user_id:
        #         each_lenth_com_round[clt_this_comm_round.index(sup_idx)] *= args.w_mul_times

        # total_lenth_com_round = sum(each_lenth_com_round)
        print('=========每轮通信数据量分布=========')
        print(each_lenth_com_round)
        sup_length = 0
        unsup_length = 0
        clt_freq_this_com_round = []
        for i in supervised_user_id:
            sup_length += each_lenth_com_round[i]
        for i in unsupervised_user_id:
            unsup_length += each_lenth_com_round[i]
        for i in supervised_user_id:
            if unsup_length == 0:
                clt_freq_this_com_round.append(each_lenth_com_round[i] / sup_length)
            else:
                clt_freq_this_com_round.append(each_lenth_com_round[i] / sup_length * 0.5)
        for i in unsupervised_user_id:
            clt_freq_this_com_round.append(each_lenth_com_round[i] / unsup_length * 0.5)
        print(clt_freq_this_com_round)
        # # 每个客户端数据量占比
        # clt_freq_this_com_round = [i / total_lenth_com_round for i in each_lenth_com_round]

        # 计算全局模型
        with torch.no_grad():
            w_glob = FedAvg(w_com_round_local, clt_freq_this_com_round)

        # 全局模型更新
        net_glob.load_state_dict(w_glob)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w_glob)
        for i in unsupervised_user_id:
            unsup_net_locals[i - sup_num].load_state_dict(w_glob)

        # 全局阈值更新，freq_global_confi为最大预测值平均
        print("==============每个客户端最大平均预测值：==============")
        print(pred_clis)
        freq_global_confi = np.mean(pred_clis)
        global_confi = global_confi_update(global_confi, args.ema_decay, com_round, freq_global_confi)
        print('====================全局阈值====================')
        print(global_confi)

        # 预测值期望更新
        mask = np.array(local_pred_exp_next) != 0
        # 计算每列非零元素的数量
        non_zero_counts = np.sum(mask, axis=0)
        # 计算每列非零元素的总和
        non_zero_sums = np.sum(local_pred_exp_next * mask, axis=0)

        local_pred_exp_next = np.mean(local_pred_exp_next, axis=0)
        # 规约到1
        # local_pred_exp_next = local_pred_exp_next / np.sum(local_pred_exp_next)
        print("======================类别概率=======================")
        print(local_pred_exp_next)
        print("===============每个类别数量================")
        print(num_class)
        # 之前的只计算不为0的项
        # local_pred_exp_next = np.divide(non_zero_sums, non_zero_counts, where=non_zero_counts != 0)

        local_pred_exp = local_pred_exp_cli_update(local_pred_exp, local_pred_exp_next, args.ema_decay, com_round)
        # print("local_pred_exp====>{}".format(local_pred_exp))

        # 计算标准差
        std_local_pred_exp = np.std(np.array(local_pred_exp))

        # 本地阈值更新
        # local_confi = global_confi + np.array(local_pred_exp) - std_local_pred_exp
        local_confi = global_confi * (np.array(local_pred_exp) / np.max(np.array(local_pred_exp)))

        loss_avg = sum(loss_locals) / len(loss_locals)
        num_class = np.array(num_class)

        # 类别权重更新
        class_weight = (1 - args.beta_c) / (1 - (args.beta_c ** num_class))
        class_weight[np.isinf(class_weight)] = 0
        # TODO
        # 权重归一化和为类别数
        zero_counts = np.sum(class_weight == 0, axis=1)
        total_sum = np.sum(class_weight, axis=1)
        scale_factor = (n_classes - zero_counts) / total_sum
        # scale_factor = 1 / total_sum
        scale_factor = np.array(scale_factor)[:, np.newaxis]
        class_weight = class_weight * scale_factor
        class_weight[class_weight == 0] = 1
        print("==============类别权重=============")
        print(class_weight)
        logger.info(
            '************ Training Loss {}, LR {}, Round {} ends ************  '.format(loss_avg, args.base_lr,
                                                                                        com_round))
        # 每6轮通信保存模型参数
        if com_round % 10 == 0:
            if not os.path.isdir(snapshot_path + time_current):
                os.mkdir(snapshot_path + time_current)
            save_mode_path = os.path.join(snapshot_path + time_current, 'epoch_' + str(com_round) + '.pth')
            if len(args.gpu) != 1:
                torch.save({
                    'state_dict': net_glob.module.state_dict(),
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round,
                    'global_confi': global_confi,
                    'local_confi': local_confi,
                    'local_pred_exp': local_pred_exp
                }
                    , save_mode_path
                )
            else:
                torch.save({
                    'state_dict': net_glob.state_dict(),
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round,
                    'global_confi': global_confi,
                    'local_confi': local_confi,
                    'local_pred_exp': local_pred_exp
                }
                    , save_mode_path
                )

        AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
        writer.add_scalar('AUC', AUROC_avg, global_step=com_round)
        writer.add_scalar('Acc', Accus_avg, global_step=com_round)
        logger.info("\nTEST Student: Epoch: {}".format(com_round))
        logger.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}"
                    .format(AUROC_avg, Accus_avg))
