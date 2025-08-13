import torch
from functools import reduce
from typing import Sequence, Tuple
from random import random
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


def label_guessing(model: nn.Module, batches, model_type=None) -> Tensor:
    model.eval()
    with torch.no_grad():
        # 一个批次数据的类别概率
        probs = F.softmax(model(batches, model=model_type)[2], dim=1)
        # reduce不支持tensor，reduce_sum才支持，因此下面这一行不起作用
        # mean_prob = reduce(lambda x, y: x + y, probs) / len(batches_1)
    # 返回的还是预测值
    return probs


def sharpen(x: Tensor, t=0.5) -> Tensor:
    sharpened_x = x ** (1 / t)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


class RandomAugmentation(nn.Module):
    def __init__(self, augmentation: nn.Module, p: float = 0.5, same_on_batch: bool = False):
        super().__init__()

        self.prob = p
        self.augmentation = augmentation
        self.same_on_batch = same_on_batch

    def forward(self, images: Tensor) -> Tensor:
        is_batch = len(images) < 4

        if not is_batch or self.same_on_batch:
            if random() <= self.prob:
                out = self.augmentation(images)
            else:
                out = images
        else:
            out = self.augmentation(images)
            batch_size = len(images)

            # get the indices of data which shouldn't apply augmentation
            indices = torch.where(torch.rand(batch_size) > self.prob)
            out[indices] = images[indices]

        return out
