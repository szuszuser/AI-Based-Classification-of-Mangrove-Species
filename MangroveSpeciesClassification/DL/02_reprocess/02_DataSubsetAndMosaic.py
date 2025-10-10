from osgeo import gdalconst, gdal
import numpy as np
import os

def crop_image_and_save(input_image, output_folder, window_size=128, stride=128):
    dataset = gdal.Open(input_image, gdalconst.GA_ReadOnly)
    if dataset is None:
        print("无法打开影像文件")
        return

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band_count = dataset.RasterCount
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    # 创建存储裁剪后影像的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 1  # 用于计数裁剪后的影像文件
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # 确保裁剪窗口在图像范围内
            if x + window_size <= width and y + window_size <= height:
                # 读取每个波段的数据
                bands_data = []
                for i in range(1, band_count + 1):
                    band = dataset.GetRasterBand(i)
                    data = band.ReadAsArray(x, y, window_size, window_size)
                    bands_data.append(data)

                # 将裁剪后的数据存储为一个影像
                cropped_image = np.stack(bands_data, axis=-1)

                # 构建保存裁剪后影像的文件路径
                output_filename = os.path.join(output_folder, f"cropped_image_{count}.tif")

                # 保存裁剪后的影像
                driver = gdal.GetDriverByName('GTiff')
                output_dataset = driver.Create(output_filename, window_size, window_size, band_count, gdalconst.GDT_Float32)
                output_dataset.SetProjection(projection)
                output_dataset.SetGeoTransform((geotransform[0] + x * geotransform[1], geotransform[1], 0, geotransform[3] + y * geotransform[5], 0, geotransform[5]))
                for i in range(band_count):
                    output_dataset.GetRasterBand(i + 1).WriteArray(cropped_image[:, :, i])
                output_dataset = None  # 关闭输出影像

                count += 1

    dataset = None  # 关闭影像文件

def crop_image_into_nine_parts_and_save(input_image, output_folder):
    dataset = gdal.Open(input_image, gdalconst.GA_ReadOnly)
    if dataset is None:
        print("无法打开影像文件")
        return

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band_count = dataset.RasterCount
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    # 创建存储裁剪后影像的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    window_width = width // 3
    window_height = height // 3
    count = 1  # 用于计数裁剪后的影像文件
    for y in range(0, 3):
        for x in range(0, 3):
            start_x = x * window_width
            end_x = start_x + window_width if x < 2 else width
            start_y = y * window_height
            end_y = start_y + window_height if y < 2 else height

            # 读取每个波段的数据
            bands_data = []
            for i in range(1, band_count + 1):
                band = dataset.GetRasterBand(i)
                data = band.ReadAsArray(start_x, start_y, end_x - start_x, end_y - start_y)
                bands_data.append(data)

            # 将裁剪后的数据存储为一个影像
            cropped_image = np.stack(bands_data, axis=-1)

            # 构建保存裁剪后影像的文件路径
            output_filename = os.path.join(output_folder, f"cropped_image_{count}.tif")

            # 保存裁剪后的影像
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.Create(output_filename, end_x - start_x, end_y - start_y, band_count, gdalconst.GDT_Float32)
            output_dataset.SetProjection(projection)
            output_dataset.SetGeoTransform((geotransform[0] + start_x * geotransform[1], geotransform[1], 0, geotransform[3] + start_y * geotransform[5], 0, geotransform[5]))
            for i in range(band_count):
                output_dataset.GetRasterBand(i + 1).WriteArray(cropped_image[:, :, i])
            output_dataset = None  # 关闭输出影像

            count += 1

    dataset = None  # 关闭影像文件


def concatenate_images_from_folder_and_save(input_folder, output_file):
    # 获取文件夹中所有.tif格式的影像文件，并根据数字排序
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith('.tif')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 根据文件名中的数字排序

    # 打开第一块影像获取相关信息
    first_image = gdal.Open(os.path.join(input_folder, files[0]), gdalconst.GA_ReadOnly)
    if first_image is None:
        print("无法打开影像文件")
        return

    width = first_image.RasterXSize
    height = first_image.RasterYSize
    projection = first_image.GetProjection()
    geotransform = first_image.GetGeoTransform()
    band_count = first_image.RasterCount

    # 确定拼接后的影像大小
    num_images = len(files)
    cols = 33  # 每行33张影像
    rows = int(np.ceil(num_images / cols))

    # 获取裁剪后影像的大小和波段数量
    image = gdal.Open(os.path.join(input_folder, files[0]), gdalconst.GA_ReadOnly)
    band = image.GetRasterBand(1)
    band_height = band.YSize
    band_width = band.XSize

    # 创建一个存储拼接后影像的数组
    concatenated_image = np.zeros((band_height * rows, band_width * cols, band_count), dtype=np.float32)

    # 将裁剪后的影像逐个填入拼接后的影像中
    for i in range(num_images):
        row = i // cols
        col = i % cols
        start_row = row * band_height
        start_col = col * band_width
        end_row = start_row + band_height
        end_col = start_col + band_width

        image_part = gdal.Open(os.path.join(input_folder, files[i]), gdalconst.GA_ReadOnly)
        for b in range(band_count):
            band_data = image_part.GetRasterBand(b + 1).ReadAsArray()
            concatenated_image[start_row:end_row, start_col:end_col, b] = band_data

    # 保存拼接后的影像
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, band_width * cols, band_height * rows, band_count, gdalconst.GDT_Float32)
    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)
    for i in range(band_count):
        output_dataset.GetRasterBand(i + 1).WriteArray(concatenated_image[:, :, i])
    output_dataset = None  # 关闭输出影像


if __name__ == '__main__':
    mode = ''
    if mode == 's':
        # subset
        input_image = r''  # 输入影像路径
        output_folder = r'' # 输入结果路径
        crop_image_and_save(input_image, output_folder)
        # crop_image_into_nine_parts_and_save(input_image, output_folder)
    else:
        # 拼接
        input_folder = r''  # 输入文件夹目录
        output_file = r''  # 输出路径文件夹
        concatenate_images_from_folder_and_save(input_folder=input_folder, output_file=output_file)
