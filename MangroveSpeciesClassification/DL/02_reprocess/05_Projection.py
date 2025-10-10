from osgeo import gdal

dat_path = r''  # 原始带坐标系影像数据
dat_dataset = gdal.Open(dat_path)
dat_width = dat_dataset.RasterXSize  # 栅格矩阵的列数
dat_height = dat_dataset.RasterYSize  # 栅格矩阵的行数

dat_geotrans = dat_dataset.GetGeoTransform()  # 仿射矩阵
dat_proj = dat_dataset.GetProjection()  # 地图投影信息
tif_path = r''  # 需要添加坐标系数据
tif_dataset = gdal.Open(tif_path, gdal.GA_Update)
tif_width = tif_dataset.RasterXSize  # 栅格矩阵的列数
tif_height = tif_dataset.RasterYSize  # 栅格矩阵的行数
tif_dataset.SetGeoTransform(dat_geotrans)
tif_dataset.SetProjection(dat_proj)
# 保存新的.tif文件
new_tif_path = r''    # 结果输出路径
driver = gdal.GetDriverByName('GTiff')
driver.CreateCopy(new_tif_path, tif_dataset)
