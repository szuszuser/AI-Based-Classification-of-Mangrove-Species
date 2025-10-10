import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset, random_split
import torchvision.transforms
import argparse
import random
from glob import glob
from abc import ABC
from osgeo import gdal
from PIL import Image


def read_img(file_name):
    """
    读取遥感数据文件
    :param file_name: 文件路径
    :return: im_proj, im_geotrans, im_data, im_width, im_height
    """
    dataset = gdal.Open(file_name)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    im_data[np.isnan(im_data)] = 0
    # im_data[im_data > np.finfo('float64').max / 2] = 0
    # im_data[im_data > np.finfo('float32').max] = 0  # 或者其他合适的值


    del dataset  # 关闭对象，文件dataset
    return im_data



class UNetDataset(Dataset, ABC):
    def __init__(self, mask_list=None, phase='train'):
        super(UNetDataset, self).__init__()
        self.phase = phase
        # get image name
        if phase != "test":
            assert mask_list, 'mask list must given when training'
            self.mask_file_list = mask_list
            self.img_file_list = [path.replace('mask', 'image') for path in mask_list]
            # self.img_file_list = [path.replace('png', 'tif') for path in self.img_file_list]
            assert len(self.img_file_list) == len(self.mask_file_list)

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        img_name = self.img_file_list[idx]
        img = read_img(img_name)

        mask_name = self.mask_file_list[idx]
        # mask = cv2.imread(mask_name, 0)
        mask = read_img(mask_name)
        mask = np.expand_dims(mask, 0)

        return torch.from_numpy(img).to(torch.float32), torch.LongTensor(mask)


if __name__ == '__main__':
    mask_path = glob(r'\*.tif')  # 标签文件夹目录

    random.seed(42)
    random.shuffle(mask_path)

    dataset = UNetDataset(mask_list=mask_path, phase='train')
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )
    print(len(dataset))
    # count = 0.0
    # pos = 0.0
    for i, (data, target) in enumerate(data_loader):
        print('i', i)
        print(data.shape)
        print(target.shape)
