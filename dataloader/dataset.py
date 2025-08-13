# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

N_CLASSES = 10


# 在原来数据的基础上进行处理，构建自己的数据集
class CheXpertDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:
            dataset_type: 数据集类型 cifar100
            data_np: 数据
            lab_trans：对有标签数据的处理步骤transforms
        """
        super(CheXpertDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        # 训练集处理
        if not is_testing:
            if is_labeled:
                self.transform = lab_trans
            # 无标签数据处理
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans_wk
        # 测试集处理
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    # 用于访问单个样本
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        elif self.dataset_type == 'fmnist':
            image = Image.fromarray(self.images[index].numpy()).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        # 转换为40*40
        image_resized = self.resize(image)
        label = self.labels[index]
        # 训练
        if not self.is_testing:
            if self.is_labeled:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]
                    # 增强的两份数据
                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                    return index, weak_aug, torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])

    # 用于返回数据集的大小
    def __len__(self):
        return len(self.labels)


# 输出两次增强数据
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return [out1, out2]
