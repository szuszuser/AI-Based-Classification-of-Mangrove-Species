import numpy as np
import gdal
import pickle
from sklearn.preprocessing import StandardScaler

# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def pre_class(XGboost_model_path, image_path, save_path):
    # Tif_bands, Tif_geotrans, Tif_proj, data_array = read_envi(image_path)
    # 调用保存好的模型
    # 以读二进制的方式打开文件
    file = open(XGboost_model_path, "rb")
    # 把模型从文件中读取出来
    rf_model = pickle.load(file)
    # 关闭文件
    file.close()
    # 用读入的模型进行预测

    # 读取数据
    dataset = gdal.Open(image_path)
    Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
    Tif_height = dataset.RasterYSize  # 栅格矩阵的行数
    Tif_bands = dataset.RasterCount  # 波段数
    Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset.GetProjection()  # 获取投影信息
    data_array = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)

    print(data_array.shape)
    #  在测试前要调整一下数据的格式
    data = np.zeros((data_array.shape[0], data_array.shape[1] * data_array.shape[2]))
    for i in range(data_array.shape[0]):
        data[i] = data_array[i].flatten()
    data = data.swapaxes(0, 1)
    data[np.isnan(data)] = 0
    #  对调整好格式的数据进行预测
    pred = rf_model.predict(data)
    #  同样地，我们对预测好的数据调整为我们图像的格式
    pred = (pred.reshape(data_array.shape[1], data_array.shape[2])) + 1
    pred = pred.astype(np.uint8)

    #  将结果写到tif图像里
    writeTiff(pred, Tif_geotrans, Tif_proj, save_path)


XGboost_model_path = r'MLModel'
image_path = r'.dat'
save_path = r'ML_map.dat'
pre_class(XGboost_model_path, image_path, save_path)
 