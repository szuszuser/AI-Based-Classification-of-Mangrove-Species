from osgeo import gdal
import os

# 设置工作目录
input_folder = r""  # 输入文件夹目录
output_folder = r""  # 输出文件夹目录

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有TIF影像文件路径
input_images = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".tif")]

# 遍历处理每张影像
for image_path in input_images:
    # 打开影像文件
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)

    # 读取影像数据
    band = dataset.GetRasterBand(1)  # 获取第一个波段
    data = band.ReadAsArray()

    # 检查影像中的像素值是否全部为0
    if not (data == 0).all():
        # 创建输出文件路径
        output_path = os.path.join(output_folder, os.path.basename(image_path))

        # 写入非全0值的影像到新文件
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.CreateCopy(output_path, dataset)

        # 释放资源
        output_dataset = None

        print(f"已保存非全0值的影像：{output_path}")

    # 关闭输入影像文件
    dataset = None