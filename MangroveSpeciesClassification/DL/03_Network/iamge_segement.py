import glob
import os
import pickle
import sys
import time
import random
import cv2
from osgeo import gdal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import UNetDataset, read_img
from unetpp import Unet3P, Unet4P, Unet5P, UnetPP, SEUnet, ECAUnet, Unet, Segnet
from torch.utils.data import Subset, random_split
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        log_softmax = nn.LogSoftmax()
        logpt = log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def accuracy(output, target):
    # total_markup = 0
    # total_intersect = 0
    _, output = output.data.max(dim=1)
    # output[output > 0] = 1
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    # nonzero_mask = (target != 0)
    # correct_pixels = 0
    # total_pixels = 0
    # correct_pixels += ((output == target) & nonzero_mask).sum().item()
    # total_pixels += nonzero_mask.sum().item()
    # result = correct_pixels / total_pixels
    # 找到非零值的位置索引
    non_zero_indices = np.nonzero(target)[0]

    # 从output和target中取出非零位置的值
    output_non_zero = output[non_zero_indices]
    target_non_zero = target[non_zero_indices]

    # 计算准确预测的数量
    correct_predictions = np.sum(output_non_zero == target_non_zero)

    # 计算准确率
    result = correct_predictions / len(non_zero_indices) * 100 if non_zero_indices.size else 0

    # print(f"预测数组的准确率为: {result:.2f}%")
    # intersect = output * target
    # total_intersect = intersect.sum()
    # total_markup = target.sum() + output.sum()
    # result = 2 * total_intersect / total_markup
    return result


class UNetTrainer(object):
    """神经网络训练"""

    def __init__(self, start_epoch=0, save_dir='', resume='', num_classes=9, color_dim=8):
        self.net = Unet(color_dim=color_dim, num_classes=num_classes)
        self.start_epoch = start_epoch if start_epoch != 0 else 1
        self.save_dir = os.path.join(r'G:\SZU\Data_for_paper2\Journal\01_GISCIENCE\01_gaoqiao\object_unet', save_dir)  # 模型存储路径
        # self.loss = CrossEntropyLoss()
        self.loss = FocalLoss()
        self.num_classes = num_classes

        if resume:
            checkpoint = torch.load(resume)  # 加载模型
            if self.start_epoch != 0:
                self.start_epoch = checkpoint['epoch'] + 1
            if self.save_dir:
                self.save_dir = checkpoint['save_dir']
            self.net.load_state_dict(checkpoint['state_dir'])  # 加载模型

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def train(self, train_loader, var_loader, lr=0.001, weight_decay=1e-4, epochs=300, save_freq=1):
        """
        学习和训练数据
        :param train_loader: 训练数据
        :param var_loader: 验证数据
        :param lr: 学习速率
        :param weight_decay: 权值衰减
        :param epochs: 学习次数
        :param save_freq: 保存数据步长
        :return:
        """
        self.logfile = os.path.join(self.save_dir, 'log.txt')
        sys.stdout = Logger(self.logfile)  # 重定向到创建文件，后面print()打印的数据都保存到log中
        self.epochs = epochs
        self.lr = lr

        optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=weight_decay)  # Adam优化, SGD
        for epoch in range(self.start_epoch, epochs + 1):
            self.train_(train_loader, epoch, optimizer, save_freq)  # 训练模型
            self.validate_(var_loader, epoch)  # 验证模型

    def train_(self, data_loader, epoch, optimizer, save_freq):
        """
        训练
        :param data_loader: 训练数据
        :param epoch: 学习次数
        :param optimizer: 优化数据
        :param save_freq: 保存数据步长
        :return:
        """
        start_time = time.time()

        if torch.cuda.device_count() > 1:
            self.net = DataParallel(self.net)
        self.net.to(device)
        self.net.train()  # 指定当前模型是在训练，module.eval()指定当前模型为验证

        metrics = []
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = Variable(data.to(device))
            target = Variable(target.to(device))

            output = self.net(data)

            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)  # 计算损失值

            optimizer.zero_grad()
            loss_output.requires_grad_(True)
            loss_output.backward()  # 反向传播
            optimizer.step()  # 权值更新

            loss_output = loss_output.item()
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])
        if epoch % save_freq == 0:
            if 'module' in dir(self.net):  # 返回net的属性
                state_dict = self.net.module.state_dict()
            else:
                state_dict = self.net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({  # 保存模型数据
                'epoch': epoch,
                'save_dir': self.save_dir,
                'state_dir': state_dict
            }, os.path.join(self.save_dir, 'UNet%03d.txt' % epoch))

        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        writer.add_scalar('loss', np.mean(metrics[:, 0]), epoch)
        writer.add_scalar('accuracy', np.mean(metrics[:, 1]), epoch)
        writer.close()
        self.print_metrics(metrics, 'Train', end_time - start_time, epoch)

    def validate_(self, data_loader, epoch):
        """
        验证
        :param data_loader: 验证数据
        :param epoch: 学习次数
        :return:
        """
        start_time = time.time()

        self.net.eval()  # 当前为验证模型
        metrics = []
        for i, (data, target) in enumerate(data_loader):
            with torch.no_grad():
                data = Variable(data.to(device))
                target = Variable(target.to(device))

            output = self.net(data)
            output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
            target = target.view(-1)
            loss_output = self.loss(output, target)

            loss_output = loss_output.item()
            acc = accuracy(output, target)
            metrics.append([loss_output, acc])
        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        writer.add_scalar('val_accuracy', np.mean(metrics[:, 1]), epoch)
        writer.close()
        self.print_metrics(metrics, 'Validation', end_time - start_time)

    def print_metrics(self, metrics, phase, time, epoch=-1):
        """
        保存指标
        :param metrics: 损失值和正确比例
        :param phase: 数据类型，分为验证和训练类型
        :param time: 保存一次数据所用时间
        :param epoch:
        :return:
        """
        if epoch != -1:
            print('Epoch: {}'.format(epoch))
        print(phase)
        print('loss %2.4f, accuracy %2.4f, time %2.2f' % (np.mean(metrics[:, 0]), np.mean(metrics[:, 1]), time))
        if phase != 'Train':
            print()


class UNetTester(object):
    """测试数据"""

    def __init__(self, model_path, target_path, color_dim=8, num_classes=9):
        self.net = Unet(color_dim=color_dim)
        checkpoint = torch.load(model_path)
        self.target_dir = target_path
        self.color_dim = color_dim
        self.num_classes = num_classes
        self.net.load_state_dict(checkpoint['state_dir'])
        self.net.to(device)
        self.net.eval()

    def test(self, testImage_dir):
        cracks_files = glob.glob(os.path.join(testImage_dir, '*.tif'))
        print(len(cracks_files), 'imgs.')
        for cracks_file in tqdm(cracks_files):  # 显示读取进度条
            name = os.path.basename(cracks_file)
            # name = name.replace('dat', 'png')
            save_path = os.path.join(self.target_dir, name)
            data = gdal.Open(cracks_file)  # 打开文件

            im_width = data.RasterXSize  # 栅格矩阵的列数
            im_height = data.RasterYSize  # 栅格矩阵的行数

            im_geotrans = data.GetGeoTransform()  # 仿射矩阵
            im_proj = data.GetProjection()  # 地图投影信息
            im_data = data.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
            im_data[np.isnan(im_data)] = 0

            # data = data.transpose(2, 0, 3, 1)
            # data = data[190:830, 190:830]
            # data = read_img(cracks_file)
            output = self._test(im_data)

            driver = gdal.GetDriverByName('Gtiff')
            dataset = driver.Create(save_path, im_width, im_height, 1, gdal.GDT_Byte)
            dataset.SetProjection(im_proj)
            dataset.SetGeoTransform(im_geotrans)
            dataset.GetRasterBand(1).WriteArray(output)

            # cv2.imwrite(save_path, output)

    def _test(self, data):
        data = data.astype(np.float32)
        data = np.expand_dims(data, 0)
        input = torch.from_numpy(data)
        height = input.size()[-2]
        width = input.size()[-1]
        with torch.no_grad():
            input = Variable(input.to(device))

        output = self.net(input)
        output = output.transpose(1, 3).transpose(1, 2).contiguous().view(-1, self.num_classes)
        _, output = output.data.max(dim=1)
        # output[output == 0] = 0
        # output[output == 1] = 30
        output = output.view(height, width)
        output = output.cpu().numpy()
        return output


if __name__ == '__main__':
    # mode = input('输入网络模式，t表示训练模式， 其他表示测试模式:')
    # 可视化损失值和准确率
    mode = ''
    writer = SummaryWriter('../object_UNet')
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'using device: {device}')
    if mode == 't':
        mask_path = glob.glob(r'\*.tif')  # 标签文件夹目录
        random.seed(1)
        random.shuffle(mask_path)
        N = int(len(mask_path) * 0.7)
        train_data = UNetDataset(mask_list=mask_path[:N], phase="train")
        val_data = UNetDataset(mask_list=mask_path[N:], phase="train")
        train_data = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
        val_data = DataLoader(val_data, batch_size=8, shuffle=True, num_workers=0)
        crack_segment = UNetTrainer(save_dir="object_UNet", resume='', color_dim=8)
        crack_segment.train(train_data, val_data)
    else:
        crack_testNet = UNetTester(
            model_path=r'', color_dim=8,
            target_path=r'')
        crack_testNet.test(testImage_dir=r'')    # 模型选择路径、结果输出路径、输入影像路径
