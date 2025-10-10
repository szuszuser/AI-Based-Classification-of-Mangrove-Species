from osgeo import gdal, gdalconst
import numpy as np

# 打开遥感影像文件
input_image_path = r''  # 替换为你的影像文件路径
output_image_path = r''  # 替换为输出路径
dataset = gdal.Open(input_image_path, gdalconst.GA_ReadOnly)

if dataset is None:
    print("无法打开影像文件")
    exit()

# 获取影像的列数和波段数
num_cols = dataset.RasterXSize
num_bands = dataset.RasterCount
num_rows = dataset.RasterYSize  # 获取影像的行数

# 计算新的影像大小
# new_num_cols = num_cols + 1
# new_num_rows = num_rows + 1
new_num_cols = num_cols + 46
new_num_rows = num_rows + 14
# 创建一个新的影像文件
driver = gdal.GetDriverByName('ENVI')  # 根据你的数据格式选择合适的驱动器
out_data = driver.Create(output_image_path, new_num_cols, new_num_rows, num_bands, gdalconst.GDT_Float32)  # 添加行和列

if out_data is None:
    print("无法创建输出影像文件")
    exit()

# 将原始数据写入新的影像文件
for i in range(1, num_bands + 1):
    band = dataset.GetRasterBand(i)
    out_band = out_data.GetRasterBand(i)
    data = band.ReadAsArray()

    # 在新的影像中写入原始数据
    out_band.WriteArray(data)

# 关闭数据集
dataset = None
out_data = None

print("影像处理完成！")
