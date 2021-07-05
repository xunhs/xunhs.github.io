---
layout: post
cid: 148
title: fmm-Map Matching路网匹配工具环境搭建与使用
slug: 148
date: 2020-03-17T09:20:26+00:00
status: publish
author: Ethan
toc: true
categories:
  - 收藏
tags:
  - Map Matching
  - 路网匹配
---


> [fmm](https://github.com/cyang-kth/fmm) 是在github上找到的比较好用的Map Matching路网匹配工具。 


<!--more-->

目录

[TOC]

### 环境搭建 (Installing)
- 建议在Ubuntu系统下搭建（笔者尝试Window 10 wsl2, cwing和Ubuntu 18.04均为成功，最终在Ubuntu16.04下编译成功）,docker ubuntu 16.04安装成功(2020.5.20)
- 主要参考[FMM-wiki](https://fmm-wiki.github.io/)
- 项目打包: [链接1](https://pan.baidu.com/s/15n-3XyP75NU7NCGicdvWFw), 提取码: 9ksw;

#### Install requirements

- 添加ppa，更新gdal相关的库
`sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt update`

P.S.: ppa包比较难下载，建议终端挂代理: `export http_proxy=http://192.168.0.11:2802`

- 安装库
`sudo apt-get install libboost-dev libboost-serialization-dev gdal-bin libgdal-dev make cmake`

#### Install C++ program

- 编译

```bash
# Under the project folder
mkdir build
cd build
cmake ..
make
sudo make install

# cmake output:
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found GDAL: /usr/lib/libgdal.so (Required is at least version "2.2") 
-- GDAL headers found at /usr/include/gdal
-- GDAL library found at /usr/lib/libgdal.so
-- Boost version: 1.58.0
-- Found the following Boost libraries:
--   serialization
-- Boost headers found at /usr/include
-- Boost library found at /usr/lib/x86_64-linux-gnu/libboost_serialization.so
-- Try OpenMP C flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Try OpenMP CXX flag = [-fopenmp]
-- Performing Test OpenMP_FLAG_DETECTED
-- Performing Test OpenMP_FLAG_DETECTED - Success
-- Found OpenMP: -fopenmp  
-- OpenMP_CXX_LIBRARIES found at 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/only/Projects/fmm-master/build

```
  
  
  
- Verfication of installation
`fmm`:

```bash
------------ Fast map matching (FMM) ------------
------------     Author: Can Yang    ------------
------------   Version: 2020.01.31   ------------
------------     Applicaton: fmm     ------------
A configuration file is given in the example folder
Run `fmm config.xml` or with arguments
fmm argument lists:
--ubodt (required) <string>: Ubodt file name
--network (required) <string>: Network file name
--gps (required) <string>: GPS file name
--output (required) <string>: Output file name
--network_id (optional) <string>: Network id name (id)
--source (optional) <string>: Network source name (source)
--target (optional) <string>: Network target name (target)
--gps_id (optional) <string>: GPS id name (id)
--gps_geom (optional) <string>: GPS geometry name (geom)
--candidates (optional) <int>: number of candidates (8)
--radius (optional) <double>: search radius (300)
--error (optional) <double>: GPS error (50)
--pf (optional) <double>: penalty factor (0)
--log_level (optional) <int>: log level (2)
--output_fields (optional) <string>: Output fields
  opath,cpath,tpath,ogeom,mgeom,pgeom,
  offset,error,spdist,tp,ep,all
For xml configuration, check example folder
------------    Program finished     ------------
```


#### Install python extension
在Python2使用

- Swig installation
	- `sudo apt-get install build-essential libpcre3-dev libpcre3`
	- Build swig  
	
```bash
tar -xf swig-4.0.1.tar.gz
cd swig-4.0.1/
./configure
sudo make
sudo make install

swig -version
```
- python-dev: `sudo apt-get install  python-dev`
- Installation of fmm Python API
编译:  
```bash
cd python
mkdir build
cd build
cmake ..
make
```
Add the `build` folder to the environment variable `PYTHONPATH`:

```bash
echo 'export PYTHONPATH=${PYTHONPATH}:PATH_TO_BUILD_FOLDER' >> ~/.bashrc
source ~/.bashrc
```
PATH_TO_BUILD_FOLDER: /workspace/fmm-master/python/build
- 验证

```bash
cd ..
python2 fmm_test.py
 ```

### 使用

样例文件打包:[链接1](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/usr/uploads/2020/03/3685900204.zip)

#### 路网数据准备

- from osm-based [osmnx](https://github.com/gboeing/osmnx)

Download by place name

```python
import osmnx as ox
place ="Stockholm, Sweden"
G = ox.graph_from_place(place, network_type='drive',which_result=2)
ox.save_graph_shapefile(G, filename='stockholm')
```

Download by a boundary polygon in geojson

```python
import osmnx as ox
from shapely.geometry import shape
json_file = open("stockholm_boundary.geojson")
import json
data = json.load(json_file)
boundary_polygon = shape(data["features"][0]['geometry'])
G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
ox.save_graph_shapefile(G, filename='stockholm')
```

- 自定义数据
假设手头上有的路网数据仅有id和geometry字段，首先我们需要构建路网拓扑结构，使用的是arcmap+postgresql，参考[这里](https://blog.csdn.net/qq_22174779/article/details/90294813)
	- 打断路网相交线
	- postgresql+postgis安装
	- 导入shp数据至postgis db(多种方式: geopandas, psql bash command, PostGIS Shapefile and DBF Loader Exporter)
		- geopandas 参考博文`Pandas/Geopandas Tricks`中的Geopandas I/O
	- 生成路网拓扑结构

```sql
ALTER TABLE public.bjrdv2pro ADD COLUMN source integer;
ALTER TABLE public.bjrdv2pro ADD COLUMN target integer;
ALTER TABLE public.bjrdv2pro ADD COLUMN length double precision;
SELECT pgr_createTopology('public.bjrdv2pro',0.00001, 'geom', 'gid');
CREATE INDEX source_idx ON bjrdv2pro("source");
CREATE INDEX target_idx ON bjrdv2pro("target");
update bjrdv2pro set length =st_length(geom);
select * from bjrdv2pro;
```
- 建立双向拓扑(Complement bidirectional edges)
Duplicate bidirectional edges, i.e., add a reverse edge  

```python
table_name = 'bjrdv2pro'
geom_col = 'geom'
sql_str = "select * from {}".format(table_name)
bjrd_gdf = gpd.read_postgis(sql=sql_str, con=engine, geom_col=geom_col)

def reverse_coords(line_string):
	coords = list(line_string.coords)
	coords.reverse()
	return LineString(coords)

_bjrd_gdf = bjrd_gdf.copy()
_bjrd_gdf['geom'] = _bjrd_gdf.geom.apply(lambda x: reverse_coords(x))
_bjrd_gdf['source'] = bjrd_gdf.target
_bjrd_gdf['target'] = bjrd_gdf.source

concat_gdf = pd.concat([bjrd_gdf, _bjrd_gdf], axis=0).reset_index(drop=True).set_geometry('geom')
concat_gdf['gid'] = concat_gdf.index
concat_gdf.to_file('bjrdv2probidrt.shp')
```




#### 配置&运行  

配置参考：[链接](https://fmm-wiki.github.io/docs/documentation/configuration/)
- ubodt配置(Preprocessing of fmm)

```bash
ubodt_gen --network bjrdv2probidrt.shp --id gid --source source --target target --output ubodt.txt --delta 4.0 
```
生成`ubodt.txt `配置文件（类似是构建路网cache之类的）
	
- fmm配置文件
`fmm_config-bj.xml`:

```xml
<fmm_config>
<input>
	<ubodt>
		<file>./bj_example/ubodt.txt</file>
	</ubodt>
	<network>
		<file>./bj_example/bjrdv2probidrt.shp</file>
		<id>gid</id>
		<source>source</source>
		<target>target</target>
	</network>
</input>
	<parameters>
		<k>50</k>
		<r>0.01</r>
		<pf>0</pf>
		<gps_error>0.005</gps_error>
	</parameters>
</fmm_config>
```
Note: 注意单位-如果是经纬度，单位就是度，0.001约等于100m; else单位是m

- built model  

```python
from __future__ import print_function
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import os
from shapely.wkt import dumps, loads
sys.path.append('/home/only/Projects/fmm-master/python/build')
import warnings
warnings.filterwarnings('ignore')
import fmm

root_dp = Path(r'/home/only/Projects/fmm-master')
example_dp = Path(root_dp, 'example')
py_test_dp = Path(root_dp, 'python')
bj_example_dp = Path(py_test_dp, 'bj_example')

fmm_config_fp = Path(bj_example_dp, 'fmm_config-bj.xml')
bj_network_fp = Path(bj_example_dp, 'bjrdv2probidrt.shp')
traj_tst_fp = Path(bj_example_dp, 'TrajPntsTst.shp')
multi_traj_tst_fp = Path(bj_example_dp, '1140.geojson')
traj_tst_result_fp = Path(bj_example_dp, 'traj_tst_result.geojson')


# fmm model
model = fmm.MapMatcher(str(fmm_config_fp))
```

- run and save result  

```python
traj_mm_df = multi_traj_tst_gdf[['traj_id', 'geometry']]

def _mm(g):
    try:
        mrst = model.match_wkt(str(g)).mgeom
        return loads(mrst)
    except Exception as ex:
        return None

traj_mm_df['geometry'] = traj_mm_df.geometry.apply(lambda g: _mm(g))
# (可选操作)丢掉匹配度非常差的路径
traj_mm_df.dropna(inplace=True)
traj_mm_df.set_geometry('geometry').to_file(traj_tst_result_fp, driver='GeoJSON')

```



***

