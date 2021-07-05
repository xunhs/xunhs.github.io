---
title: ArcGIS Pro栅格切割及应用于Image Embeddings札记
author: Ethan
type: post
date: 2021-02-27T13:22:36+00:00
url: /2021/02/27/414/
argon_hide_readingtime:
  - 'false'
argon_meta_simple:
  - 'false'
argon_first_image_as_thumbnail:
  - default
views:
  - 10
categories:
  - 收藏
tags:
  - Embedding
  - GIS

---
> 使用ArcGIS Pro栅格切割工具做分块裁剪，并使用切割好的分块做Image Embeddings。记录操作过程及注意事项。

<!--more-->


### Raster Split
#### 数据准备
- 北京市城区栅格数据（From 水经注：Google Maps）
- 北京市五环矢量数据（研究范围）
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227161912.png)

#### 分块裁剪
- 生成外包矩形：使用北京市五环矢量数据生成外包矩形
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227162710.png)
- 掩膜提取：使用外包矩形切割栅格影像
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227164654.png)
- 分块裁剪
设置分块大小，自动裁剪影像；格式选择tif格式
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227165313.png)
	- 注意：~~ArcGIS Pro进行此操作时存在Bug（异常错误（或者已完成，无结果）），**在环境变量中设置并行参数为0即可**~~ ArcGIS Pro更新后没有出现该问题
	![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227165629.png)
	- 生成文件如下：
	![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210420154615.png)

#### 后处理及获取中心坐标
- 获取中心坐标，便于后续空间关联操作
```python
	from osgeo import gdal
	import geopandas as gpd 
	import pandas as pd 
	from pathlib import Path
	import os 
	from tqdm import tqdm
	
	patch_size = 256
	split_raster_dir = Path('/workspace', 'ImagePatchEmbedding', 'SplitingPatch', 'BJR5_256')
	all_tif = [_ for _ in split_raster_dir.glob('*.TIF')]
	for tif_path in tqdm(all_tif):
		ds = gdal.Open(str(tif_path))
		width = ds.RasterXSize
		height = ds.RasterYSize
		ds = None 
		if width != patch_size or height != patch_size: # delete
			path_stem = tif_path.stem
			path_parent = tif_path.parent
			_l = ['.tfw', '.TIF', '.TIF.aux.xml', '.TIF.ovr']
			for _ in _l:
				d_fp = Path(path_parent, f'{path_stem}{_}')
				os.remove(d_fp)
		
	_dict = {
		'rs_stem': [],
		'centralx': [],
		'centraly': [], 
	}
	all_tif = [_ for _ in split_raster_dir.glob('*.TIF')]
	for tif_path in tqdm(all_tif):
		path_stem = tif_path.stem
		ds = gdal.Open(str(tif_path))
		width = ds.RasterXSize
		height = ds.RasterYSize
		gt = ds.GetGeoTransform()
		minx = gt[0]
		miny = gt[3] + width*gt[4] + height*gt[5] 
		maxx = gt[0] + width*gt[1] + height*gt[2]
		maxy = gt[3] 
		centralx = (minx + maxx) / 2
		centraly = (miny + maxy) / 2
		_dict['rs_stem'].append(path_stem)
		_dict['centralx'].append(centralx)
		_dict['centraly'].append(centraly)
	bj_rs_split_df = pd.DataFrame(_dict)
	bj_rs_split_df.head()

	bj_arial_image_split_fp = Path(split_raster_dir.parent, 'BJR5_256_patch_location.txt')
	bj_rs_split_df.to_csv(bj_arial_image_split_fp, header=True, index=False)
```
导入ArcGIS：
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227210237.png)

- GeoTiff转rgb（JPG图片）
使用一些方式（rasterio等）转换为jpg格式后，转换后的jpg图片一片漆黑，参考[解释](https://blog.csdn.net/tuoyakan9097/article/details/81261675?utm_medium=distribute.pc_relevant_bbs_down.none-task--2~all~sobaiduend~default-1.nonecase&depth_1-utm_source=distribute.pc_relevant_bbs_down.none-task--2~all~sobaiduend~default-1.nonecase)。需进行rgb转换，参考[方法](https://blog.csdn.net/weixin_43490758/article/details/114753913)：
```python
	# -*- coding: UTF-8 -*-
	import numpy as np
	import os
	from PIL import Image
	from osgeo import gdal
	from pathlib import Path
	from tqdm import tqdm

	def readTif(imgPath, bandsOrder=[1,2,3]):
		"""
		读取GEO tif影像的前三个波段值，并按照R.G.B顺序存储到形状为【原长*原宽*3】的数组中
		:param imgPath: 图像存储全路径
		:param bandsOrder: RGB对应的波段顺序，如高分二号多光谱包含蓝，绿，红，近红外四个波段，RGB对应的波段为3，2，1
		:return: R.G.B三维数组
		"""
		dataset = gdal.Open(imgPath, gdal.GA_ReadOnly)
		cols = dataset.RasterXSize
		rows = dataset.RasterYSize
		data = np.empty([rows, cols, 3], dtype=float)
		for i in range(3):
			band = dataset.GetRasterBand(bandsOrder[i])
			oneband_data = band.ReadAsArray()
			data[:, :, i] = oneband_data
		return data

	def stretchImg(imgPath, resultPath, lower_percent=0.5, higher_percent=99.5):
		"""
		#将光谱DN值映射至0-255，并保存
		:param imgPath: 需要转换的tif影像路径（***.tif）
		:param resultPath: 转换后的文件存储路径(***.jpg)
		:param lower_percent: 低值拉伸比率
		:param higher_percent: 高值拉伸比率
		:return: 无返回参数，直接输出图片
		"""
		RGB_Array=readTif(imgPath)
		band_Num = RGB_Array.shape[2]
		JPG_Array = np.zeros_like(RGB_Array, dtype=np.uint8)
		for i in range(band_Num):
			minValue = 0
			maxValue = 255
			#获取数组RGB_Array某个百分比分位上的值
			low_value = np.percentile(RGB_Array[:, :,i], lower_percent)
			high_value = np.percentile(RGB_Array[:, :,i], higher_percent)
			temp_value = minValue + (RGB_Array[:, :,i] - low_value) * (maxValue - minValue) / (high_value - low_value)
			temp_value[temp_value < minValue] = minValue
			temp_value[temp_value > maxValue] = maxValue
			JPG_Array[:, :, i] = temp_value
		outputImg = Image.fromarray(np.uint8(JPG_Array))
		outputImg.save(resultPath)

	def Batch_Convert_tif_to_jpg(imgdir, savedir):
		#获取文件夹下所有tif文件名称，并存入列表
		all_tif = [_ for _ in split_raster_dir.glob('*.TIF')]
		for tif in tqdm(all_tif):
			stretchImg(str(tif), str(tif.with_suffix('.jpg')))
		print("完成所有图片转换!")

	Batch_Convert_tif_to_jpg(split_raster_dir, split_raster_dir)
```


- 删除多余文件
```python
	# del extra files, only keep jpg
	_del = ['.tfw', '.TIF', '.TIF.aux.xml', '.TIF.ovr']
	all_jpg = [_ for _ in split_raster_dir.glob('*.jpg')]
	for jpg in tqdm(all_jpg):
		for _p in _del:
			os.remove(jpg.with_suffix(_p))
```


### Image Embeddings
Image Embedding初衷是使用Embedding的方式对图像进行表示，后续可应用于图片检索等应用。此处参考(https://github.com/rom1504/image_embeddings)实现该功能。在本机配置环境的过程中，faiss这个包始终报错，且网上并无解决方案。故使用Google Calab在线Jupyter Notebook的平台进行操作。

#### 数据准备
- 压缩数据并上传至Google 云端硬盘
- 将作者分享的`using_the_lib.ipynb`保存至自己的Google云端硬盘
- 安装repo: `!pip install -U image_embeddings`
- 挂在Google云端硬盘
	```Python
		from google.colab import drive
		drive.mount('/content/drive')
	```
- 解压：`!unzip "/content/drive/MyDrive/DataUploads/ImageEmbeddings/RasterSplitJPG.zip" -d "/content/drive/MyDrive/DataUploads/ImageEmbeddings/"`
- 载入repo及路径：
	```Python
		# Let's define some paths where to save images, tfrecords and embeddings
		from pathlib import Path
		import image_embeddings
		home = Path("/content/drive/MyDrive/")
		path_images = str(Path(home, "DataUploads/ImageEmbeddings/RasterSplitJPG/"))
		path_tfrecords = str(Path(home, "Colab Notebooks/ImageEmbeddings/tfrecords"))
		path_embeddings = str(Path(home, "Colab Notebooks/ImageEmbeddings/embeddings"))
	```
- 重命名`*.JPG`(原作者代码中操作的是*.jpeg格式，JPG可以读取，但是在后续图片显示操作中会提示路径错误)
	```Python
		# rename *.JPG with *,jpeg
		import os
		root_path = Path("/content/drive/MyDrive/DataUploads/ImageEmbeddings/RasterSplitJPG")
		for fp in root_path.glob("*.JPG"):
		  stem = fp.stem
		  os.rename(fp, Path(root_path, f"{stem}.jpeg"))
	```

#### Build embeddings
- Transform image to tf records：Tf record is an efficient format to store image, it's better to use than raw image file for inference
	```Python
		image_embeddings.inference.write_tfrecord(image_folder=path_images, output_folder=path_tfrecords, num_shards=10)
	```
- Build embeddings：Here, **efficientnet** is used, but the code is particularly simple, and any other model could be used. <u>The input is tfrecords and the output is embeddings</u>
	```Python
		# 如出现内存溢出错误，可适当降低batch size
		image_embeddings.inference.run_inference(tfrecords_folder=path_tfrecords, output_folder=path_embeddings, batch_size=500)
	```

#### Image Search
- Read the embeddings and build an index with it. **The knn index** is built using [faiss](https://github.com/facebookresearch/faiss) which makes it possible to search embeddings in log(N) with lot of options to reduce memory footprint
	```Python
		[id_to_name, name_to_id, embeddings] = image_embeddings.knn.read_embeddings(path_embeddings)
		index = image_embeddings.knn.build_index(embeddings)
	```
- Search in the index
	- example 1
		```Python
			name_to_id['BJ_5R_AI_189'] # 工业区
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			7440
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			p=7440
			print(id_to_name[p])
			image_embeddings.knn.display_picture(path_images, id_to_name[p])
			results = image_embeddings.knn.search(index, id_to_name, embeddings[p])
			image_embeddings.knn.display_results(path_images, results)
		```
		![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210228114254.png)
	- example 2
		```Python
			name_to_id['BJ_5R_AI_636'] # 绿地
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			5553
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			p=5553
			print(id_to_name[p])
			image_embeddings.knn.display_picture(path_images, id_to_name[p])
			results = image_embeddings.knn.search(index, id_to_name, embeddings[p])
			image_embeddings.knn.display_results(path_images, results)
		```
		![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210228114532.png)
	- example 3
		```Python
			name_to_id['BJ_5R_AI_1423'] # 居住区
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			3818
			>>>>>>>>>>>>>>>>>>>>>>>>>>
			p=3818
			print(id_to_name[p])
			image_embeddings.knn.display_picture(path_images, id_to_name[p])
			results = image_embeddings.knn.search(index, id_to_name, embeddings[p])
			image_embeddings.knn.display_results(path_images, results)
		```
		![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210228114703.png)
	- 效果还是蛮好的
- Combination of images：**Any vector in the same space can be used as query** For example I could have 2 image and want to find some example that are closeby to the 2, Let's just **average them** and see that happens !
	```Python
		p1 = 7440
		p2 = 5553
		image1 = id_to_name[p1]
		image2 = id_to_name[p2]
		image_embeddings.knn.display_picture(path_images, image1)
		image_embeddings.knn.display_picture(path_images, image2)
		results = image_embeddings.knn.search(index, id_to_name, (embeddings[p1] + embeddings[p2])/2, 7)
		image_embeddings.knn.display_results(path_images, results)
	```
	![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210228114940.png)
	可以检索出混合工业区和绿地的区域
- Normalize embedding: We get mostly one of the picture. **One thing that can be done to improve this is to normalize the embeddings to get a better mix**
	- Normalize
		```Python
			import numpy as np
			def normalized(a, axis=-1, order=2):
			l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
			l2[l2==0] = 1
			return a / np.expand_dims(l2, axis)
			normalized_embeddings = normalized(embeddings, 1)
			index_normalized = image_embeddings.knn.build_index(normalized_embeddings)
		```
	- Normalize Index
		```Python
			p1 = 7440
			p2 = 5553
			image1 = id_to_name[p1]
			image2 = id_to_name[p2]
			image_embeddings.knn.display_picture(path_images, image1)
			image_embeddings.knn.display_picture(path_images, image2)
			results = image_embeddings.knn.search(index_normalized, id_to_name, (normalized_embeddings[p1] + normalized_embeddings[p2])/2, 7)
			image_embeddings.knn.display_results(path_images, results)
		```
		![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210228115412.png)
		作者说正则化（normalize）embeddings后可以提高混合检索的精度，在一定程度上确实有所提高
- Exporting the embeddings to numpy
	```Python
		from image_embeddings.knn import embeddings_to_numpy
		path_embeddings_numpy = f"{home}/{dataset}/embeddings_numpy"
		embeddings_to_numpy(path_embeddings, path_embeddings_numpy)
	```


------------

![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210227212304.jpg)

