# import os
# import sys
#
# import shutil
# import argparse
# import logging
# import time
# import random
# import numpy as np
# import pandas as pd

import torch
from torch.nn import functional as F
from utils.metrics import compute_metrics_test


def epochVal_metrics_test(model, dataLoader, model_type, n_classes):
    # 当前是否处于训练模式，评估完后用来保持原状，是则开启训练，否则保持
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    # 每个数据的标签
    gt_study = {}
    # 每个数据的预测值列表
    pred_study = {}
    # 数据索引
    studies = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, feature, output = model(image, model=model_type)
            study = study.tolist()
            output = F.softmax(output, dim=1)
            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])
        # 对每个数据
        for study in studies:
            # gt二维 样本数 * 标签值(1)
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            # pred二维 样本数 * 预测值向量(100)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        # gt=F.one_hot(gt.to(torch.int64).squeeze())
        # AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus, Pre, Recall = compute_metrics_test(gt, pred, n_classes=n_classes)

    model.train(training)

    return AUROCs, Accus, Pre, Recall  # ,all_features.cpu(),all_labels.cpu()#, Senss, Specs, pre,F1
