---
title: Pandas/Geopandas Tricks
slug: pandas-geopandas-tricks
date: 2020-01-13T09:01:39.000Z
pinned: true
categories:
  - 收藏
tags:
  - Python
  - Pandas
  - GeoPandas
  - 优化
lastmod: '2021-07-06T01:21:07.606Z'
---

> 总结个人使用中常用Pandas及扩展插件使用技巧


<!--more-->



### I/O
#### pandas可以直接读取压缩文件，同样写可以写入压缩文件
[参考](https://twitter.com/justmarkham/status/1146764820697505792)
You can read directly from a compressed file, Or write to a compressed file. 
Also supported: .gz, .bz2, .xz

#### HDFStore
尽可能的避免读取原始csv，使用hdf、feather或h5py格式文件加快文件读取 ([参考1](https://zhuanlan.zhihu.com/p/81554435), [参考2](https://www.cnblogs.com/feffery/p/11135082.html))
HDF5（Hierarchical Data Formal）是用于存储大规模数值数据的较为理想的存储格式，文件后缀名为h5，存储读取速度非常快，且可在文件内部按照明确的层次存储数据，同一个HDF5可以看做一个高度整合的文件夹，其内部可存放不同类型的数据。
```Python
import pandas as pd
import numpy as np

# 创建新的对象、读入已存在的对象
store = pd.HDFStore('demo.h5')

# 导出到已存在的h5文件中，这里需要指定key
df_.to_hdf(path_or_buf='demo.h5',key='df_')


s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
df = pd.DataFrame(np.random.randn(8, 3), columns=["A", "B", "C"])
# 将 Series 或 DataFrame 存入 store
store["s"], store["df"] = s, df

# 查看 store 中有哪些数据
store.keys()
# out: ['/df', '/s']

# 取出某一数据
df = store["df"]

# 删除store对象中指定数据
del store['s']

# 将当前的store对象持久化到本地
store.close()

# 查看连接状况
store.is_open
```
HDF5用时仅为csv的1/13，因此在涉及到数据存储特别是规模较大的数据时，HDF5是你不错的选择。

#### 读取csv
```Python
one_piece_df = pd.read_csv(csv_path, header = 0, encoding='gbk', engine='python', error_bad_lines=False)
```
- encoding: 编码问题
- **engine**:  报错- ParserError: Error tokenizing data. C error: EOF inside string starting at row 15946
- **error_bad_lines: 忽略有错误的行, 这个用处比较大，有很多类型的报错都可以解决，建议一般情况下加上**: Skipping line 15513: ’ ’ expected after ‘"’; Skipping line 15546: unexpected end of data; ParserError: Expected 19 fields in line 212, saw 20field larger than field limit (131072)


#### 保存为json
```Python
# 建议保存方法:
parcels_info_df.to_json('ParcelsInfo.json', orient='index’)
# 同样读取方法：
pd.read_json('ParcelsInfo.json', orient='index’)
```
其他保存方法：
```Python
with open('./name.json', 'w', encoding='utf-8') as fp:
    json.dump(result_dict, fp, indent=4)
```

#### DataFrame and Dict:
```Python
# Dict 2 DataFrame:
kmeans_result_df = pd.DataFrame.from_dict(kmeans_result_dict)
# DataFrame 2 Dict:
kmeans_result_dict = pd.DataFrame.to_dict(kmeans_result_df)
```

#### List of dict and DataFrame
```Python
# List of dict to DataFrame
data_list = [{'points': 50, 'time': '5:00', 'year': 2010}, 
             {'points': 25, 'time': '6:00', 'month': "february"}, 
             {'points':90, 'time': '9:00', 'month': 'january'}, 
             {'points_h1':20, 'month': 'june'}]
df = pd.DataFrame(data_list)
# DataFrame to List of dict
data_list = df.T.to_dict().values()
```


#### dict2nametuple
```python
from collections import namedtuple
args_dict = {
    'no_cuda': False, 
    'fastmode': False, 
    'seed': 666, 
    'epochs': 500,
    'lr': 0.01, 
    'weight_decay': 5e-4, 
    'hidden': 64, 
    'dropout': 0.5,
}
Args = namedtuple('Args', [_ for _ in args_dict.keys()])
args = Args(**(args_dict))
```

#### DataFrame导出Markdown
```python
from tabulate import tabulate

df = DataFrame({
    "weekday": ["monday", "thursday", "wednesday"],
    "temperature": [20, 30, 25],
    "precipitation": [100, 200, 150],
}).set_index("weekday")

md = tabulate(df, tablefmt="pipe", headers="keys")
print(md)
```

#### joblib
```python
# 保存变量
metrics_fp = Path(gensim_model_dir, 'metrics.dat')
joblib.dump(value=metric_list, filename=str(metrics_fp))

# 载入变量
metrics_fp = Path(gensim_model_dir, 'metrics.dat')
# metric_list = joblib.load(metrics_fp)
```

### 数据库交互

#### postgresql交互
```python
from sqlalchemy import create_engine
from geoalchemy2 import Geometry, WKTElement

import pandas as pd
import geopandas as gpd

'''
Geopanda, pandas 2 postgresql
postgis操作在建立数据库后需添加postgis扩展，可在pgAdmin中新建数据库后添加
'''


class Transit(object):
    def __init__(self, engine_string, dbschema):
        self.engine_string = engine_string
        self.dbschema = dbschema
        self.engine = self.connect()

    def connect(self):
        engine = create_engine(
            self.engine_string,
            use_batch_mode=True,
            connect_args={'options': '-csearch_path={}'.format(self.dbschema)})
        return engine

    def list_tables(self, ):
        return self.engine.table_names()
    

    def to_dataframe(self, table_name):
        df = pd.read_sql_table(table_name, self.engine)
        return df
    
	def to_geodataframe_by_query(self, query_str, geom_col):
        gdf = gpd.read_postgis(sql=query_str, con=self.engine, geom_col=geom_col)
		return gdf
    
    def to_geodataframe(self, table_name, geom_col):
        sql_str = "select * from {}".format(table_name)
        gdf = self.to_geodataframe_by_query(self, sql_str, geom_col)
        return gdf
    
    def write_dataframe(self, df, table_name):
        df.to_sql(table_name, self.engine)


    def write_geodataframe(self, gdf, table_name, if_exists='replace', geometry_str='geometry'):
        '''
            gdf: geopandas geodataframe
            table_name: 
            if_exists: {‘fail’, ‘replace’, ‘append’}
            Geometry: See :class:`geoalchemy2.types._GISType` for the list of arguments that can
    be passed to the constructor
        '''
        gdf.to_sql(name=table_name,
                   con=self.engine,
                   if_exists=if_exists,
                   index=False,
                   dtype={geometry_str: Geometry('POINT', srid=4326)})
  
        



if __name__ == "__main__":

    # ----------------------- 定义数据库参数 -------------------------#
    # follows django database settings format, replace with your own settings
    DATABASES = {
        'db1': {
            'NAME': 'postgis',
            'USER': 'postgres',
            'PASSWORD': '123345678',
            'HOST': 'localhost',
            'PORT': 5432,
        },
    }

    # choose the database to use
    db = DATABASES['db1']

    # construct an engine connection string
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=db['USER'],
        password=db['PASSWORD'],
        host=db['HOST'],
        port=db['PORT'],
        database=db['NAME'],
    )

    # 选择特定schema保存（默认保存在public）;public一定要加在尾部（不然geometry写入时会报错），逗号不能有空格
    dbschema = 'chongqing,public' 
    # ----------------------- 定义数据库参数 -------------------------#

    # ----------------------- 列出表 -------------------------#
    transit = Transit(engine_string, dbschema)
    transit.list_tables()
    # []
    # ----------------------- 列出表 -------------------------#

    # ----------------------- 写入数据至postgresql -------------------------#
    gdf = gpd.read_file('./重庆市.geojson')
    gdf['wgs_geometry'] = gdf['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))
    gdf.drop(['geometry'], axis=1, inplace=True)
    transit.write_geodataframe(gdf, table_name, 'append', geometry_str='wgs_geometry')
    
    #------
    transit.write_dataframe(gdf[['adcode', 'name']], 'chongqing_attr')
    # ----------------------- 写入数据至postgresql -------------------------#

    # ----------------------- 从postgresql读出数据 -------------------------#
    table_name, geom_col = 'chongqing', 'geometry'
    gdf = transit.to_geodataframe(table_name,geom_col)
    gdf.plot()
    table_name = 'chongqing_attr'
    df = transit.to_dataframe(table_name)
    # ----------------------- 从postgresql读出数据 -------------------------#
```



#### mongodb交互
将 DataFrame 保存至 mongodb
```Python
mongo.collection.insert(json.loads(df.T.to_json()).values())
```

#### to_sqlite
有时候大批量的df.query查询太耗时间了，没有**sql查询速度快**。因此想到的一个解决方案是把查询目标的DataFrame存储到sqlite数据库，然后使用sql进行查询
```Python
# pip install sqlalchemy
from sqlalchemy import create_engine
import os
import geopandas as gpd
import pandas
from tqdm import tqdm


data_root = "../data"
nodes_fp = os.path.join(data_root, "wh/wh.shp/nodes.shp")
edges_fp = os.path.join(data_root, "wh/wh.shp/edges.shp")
edges_gdf = gpd.read_file(edges_fp)

# Create an in-memory SQLite database.
engine = create_engine('sqlite://', echo=False)
edges_gdf[['fid', 'from', 'to']].to_sql('edges', con=engine, if_exists='replace')

# 查询edges中fid为1，2的元素，并取值from和to
results = engine.execute("SELECT e.'from', e.'to'  FROM edges as e where e.'fid'in (1,2,3, '')").fetchall()
# results examle: 
# [(267620078, 3488417963), (3488417935, 267620078), (3684965660, 267620078)]

# list 转 tuple后查询
_list = [1,2,3]
engine.execute("SELECT e.'from', e.'to'  FROM edges as e where e.'fid'in {}".format(tuple(_list))).fetchall()
```
{{< notice success >}} 
- [df.to_sql](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html); 
- create_engine:如果想保存到本地，`sqlite://`后填入地址即可，如Windows路径：`engine = create_engine('sqlite:///C:\\sqlitedbs\\pois.db', echo=True)`及linux路径：`engine = create_engine('sqlite:////workspace/UrbanFunctionalRegionalization/20210702-wh/data/poi/pois.sqlite', echo=True)`
- 此处转换DataFrame为一个SQLite数据库，放在内存中
- if_exists : {'fail', 'replace', 'append'}, default 'fail'
  * fail: Raise a ValueError.
  * replace: Drop the table before inserting new values.
  * append: Insert new values to the existing table.
- to_sql还可以通过`dtype={"A": Integer()}`来定义数据类型，需先引入`from sqlalchemy.types import Integer`; sqlalchemy通用数据类型💨[sqlalchemy.types](https://docs.sqlalchemy.org/en/14/core/type_basics.html#generic-types)
{{< /notice >}}


### 数据处理

#### 基础操作

##### 数据筛选（行操作）
在筛选数据的时候，我们一般用`df[条件]`的格式，其中的条件，是对data每一行数据的true和false布尔变量的Series
- 条件：例如，我们想得到车牌照为22271的所有数据。首先我们要获得一个布尔变量的Series，这个Series对应的是data的每一行，如果车牌照为"粤B4H2K8"则为true，不是则为false。这样子的Series很容易获得，只需要`df['VehicleNum']==22271`
- **筛选**数据：
  - 单一条件`df[df['VehicleNum']==22271]`
  - 多条件：
    - 并：`df[(df['popularity'] > 3) & (df['popularity'] < 7)]`
    - 或：`df[(df['popularity'] < 3) | (df['popularity'] > 7)]`
  - 返回满足条件的行号(索引)：`np.where(df['VehicleNum']==22271)`
  - **提取某一行数据**：`df.iloc[32]`
  - 提取popularity列最大值所在行: `df[df['popularity'] == df['popularity'].max()]`
- 反向筛选：`data[-(条件)]`，例如: `data[-(data['VehicleNum']==22271)]`
- **添加一行**数据: `df = df.append({'grammer':'Perl','popularity':6.6},ignore_index=True)`
- **去除重复**行： `df.drop_duplicates(subset=None, keep='first', inplace=False)`
	- subset : column label or sequence of labels, optional 用来指定特定的列，默认所有列
	- keep : {‘first’, ‘last’, False}, default ‘first’ 删除重复项并保留第一次出现的项
	- inplace : boolean, default False 是直接在原来数据上修改还是保留一个副本
	- 参考: [drop_duplicates](https://blog.csdn.net/u010665216/article/details/78559091)
- 将**数据排序**,并把排序后的数据赋值给原来的数据：
```Python
df = df.sort_values(by = ['VehicleNum','Stime'], ascending = True)
#ascending: True 升序,False 降序
```
- **遍历**行: 如果必须要要用iterrows，可以用itertuples来进行替换。**在任何情况下itertuples都比iterrows快很多倍**。
  ```python
  for row in df.itertuples():
      print(getattr(row, 'c1'), getattr(row, 'c2'))
  ```

##### 获取/删除/定义DataFrame的某一列（列操作）
- 获取列'Stime'：`df['Stime']`或`df.loc[:,'Stime']`
- 删除列'Stime'：`df.drop(['Stime'],axis=1)`
- 获取某一列某一行的数据：`df['Stime'].iloc[3] #获取Stime列的第4行数据`
- 列（Columns）**重命名**：`df.rename(columns={"x": "pu_x", "y": "pu_y"}, inplace=True)`
- 某一列**类型转换**：`df['salary'].astype(np.float64)`
- 索引：
- **重置行号**：`df.reset_index()`
- 设置索引：`df.set_index('car_id')`
- **统计出现频率/次数**：例如，
- 查看每种学历出现的次数：`df.education.value_counts()`
- 查看education列共有几种学历：`df.education.nunique()`

##### 查看DataFrame基本信息
- 查看索引、数据类型和内存信息：`df.info()`
- 查看数值型列的汇总统计： `df.describe()`
- 查看df所有数据的最小值、25%分位数、中位数、75%分位数、最大值：`np.percentile(df, q=[0, 25, 50, 75, 100])`
- **EDA分析**(数据可视化): [sweetviz](https://towardsdatascience.com/sweetviz-automated-eda-in-python-a97e4cabacde)![sweetviz快速EDA示例](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/uploads/images/2020/S3/20200711175504.png)
	- Init:
```python
  import sweetviz as sz
  import pandas as pd
  df = pd.read_csv('train_set.csv', header=0)
  df1 = pd.read_csv('test_set.csv', header=0)
```
	- 综合报告:常见数据特征报告， [link](https://cdn.jsdelivr.net/gh/xunhs/image_host/assets/python/sweetviz/Advertising.html)
	```python
	advert_report = sz.analyze(df)
	advert_report.show_html('Advertising.html')
	```
	
	- 对比报告:如训练集和测试集对比， [link](https://cdn.jsdelivr.net/gh/xunhs/image_host/assets/python/sweetviz/Comparing.html)
	```python
	compare_report = sz.compare(df.drop('y', axis=1), df1)
	compare_report.show_html('Comparing.html')
	```


- **Pandas基本数据类型**dtype![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200312175149.png)[参考](https://pbpython.com/pandas_dtypes.html)


##### 缺失值
- 查看每列数据缺失值情况：`df.isnull().sum()`
- 提取日期列**含有空值的行**：`df[df.datetime.isnull()]`
- 删除存在缺失值的行：`df.dropna(axis=0, how='any', inplace=True)`
  - axis：0-行操作（默认），1-列操作
  - how：any-只要有空值就删除（默认），all-全部为空值才删除


##### 关联和合并
- 合并concat（轴向连接）（无需键值，直接合并，A和B具有相同的结构）
```python
# pd.concat([A, B]) # 有[]
pd.concat([A, B], axis=1) # 列之间拼接
pd.concat([A, B], axis=0) # 行之间拼接
```
- 关联merge（[数据库风格的合并](https://blog.csdn.net/weixin_38168620/article/details/80663892)）（需指定键值，依照键值匹配关系连接）
```python
# pd.merge(A, B, left_on, right_on, how) # 无[]
pd.merge(A, B, left_on='airport_ref', right_on='id', how='inner')
```

#### query()
参考[基于query()的高效查询](https://www.cnblogs.com/feffery/p/13440148.html)

##### 示例
找出类型为TV Show且国家不含美国的Kids' TV
![](https://cdn.jsdelivr.net/gh/xunhs/image_host/images/2020/8/1597799020.png)

##### 常用特性
- 直接解析字段名
在使用query()时我们在不需要重复书写数据框名称[字段名]这样的内容，字段名也直接可以当作变量使用，而且不同条件之间不需要用括号隔开，在条件繁杂的时候简化代码的效果更为明显。
- 链式表达式
  ```python
  demo = pd.DataFrame({
      'a': [5, 4, 3, 2, 1],
      'b': [1, 2, 3, 4, 5]
  })
  
  demo.query("a <= b != 4")
  ```
- 支持in与not in判断: `netflix.query("release_year in [2018, 2019]")`
- 对外部变量的支持:query()表达式还支持使用外部变量，只需要在外部变量前加上@符号即可
  ```python
  years = [2018, 2019]
  netflix.query("release_year in @years")
  ```
- 对常规语句的支持: 可以直接解析Python语句，极大地自由度
  ```python
  def country_count(s):
      '''
      计算涉及国家数量
      '''
      return s.split(',').__len__()
  
  # 找出发行年份在2018或2019年且合作国家数量超过5个的剧集
  netflix.query("release_year.isin([2018, 2019]) and country.apply(@country_count) > 5")
  ```
- 对Index与MultiIndex的支持

#### apply()

##### apply + lambda
```Python
data.gender.apply(lambda x:'女性' if x is 'F' else '男性')
# 等同于: data.gender.map({'F': '女性', 'M': '男性'})
```

##### apply输入多参
```Python
def _get_coordinates(row, points_df):
    return points_df[points_df.point_id.isin(row.traj_points)]
	.apply(lambda row: (row.x, row.y), axis=1).tolist()

trajs_df['coordinates'] = trajs_df.progress_apply(_get_coordinates, axis=1, args=(points_df,))
```
- 使用args输入多参数
- 函数参数列表中，**row放在第一个，其他参数向后延续**

##### apply 输入多列数据
```Python
def generate_descriptive_statement(year, name, gender, count):
    year, count = str(year), str(count)
    gender = '女性' if gender is 'F' else '男性'
    return '在{}年，叫做{}性别为{}的新生儿有{}个。'.format(year, name, gender, count)

data.apply(lambda row:generate_descriptive_statement(row['year'],
                                                      row['name'],
                                                      row['gender'],
                                                      row['count']),
           axis = 1)
```
- **axis=1** 处理多个值时要给apply()添加参数axis=1
- `row['year'], row['gender']` 直接用列名即可（`row.year, row.gender`也是可以的）


##### apply 输出多列数据
```Python
# 提取name列中的首字母和剩余部分字母
_apply = data.apply(lambda row: (row['name'][0], row['name'][1:]), axis=1)
a, b = zip(*list(_apply))
```
- zip(*zipped)来解开元组序列;同样在函数传参的过程中，`**args`也可以解开args字典变换参数形式。



##### apply + [swifter并行](https://github.com/jmcarpenter2/swifter/blob/master/docs/documentation.md)

> 2020.9.5 Note: swifter.apply加速效果很明显；读取大文件可以使用modin.pandas进行读取，apply等操作可以使用swifter进行加速。;另swifter.apply的函数中不可定义[vectorized form](https://github.com/jmcarpenter2/swifter/blob/master/examples/swifter_apply_examples.ipynb)（如if函数），否则可能导致加速效果不明显。

```Python
import pandas as pd
import swifter

df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [5, 6, 7, 8]})

# runs on single core
df['x2'] = df['x'].apply(lambda x: x**2)
# runs on multiple cores
df['x2'] = df['x'].swifter.apply(lambda x: x**2)

# use swifter apply on whole dataframe
df['agg'] = df.swifter.apply(lambda x: x.sum() - x.min())

# use swifter apply on specific columns
df['outCol'] = df[['inCol1', 'inCol2']].swifter.apply(my_func)
```


#### 时间处理

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200601153141.png)
参考: [pandas.pydata](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#)
另见[datetime时间处理](https://xunhs.press/2020/04/5cdaf520/#%E6%97%B6%E9%97%B4%E5%A4%84%E7%90%86)


##### parse_dates
在 `read_csv()` 方法中，通过 parse_dates 参数直接将某些列转换成 datetime64 类型, index_col设置索引
```python
df1 = pd.read_csv('sample-salesv3.csv', parse_dates=['date'], index_col='date')
```

##### to_datetime

###### Timestamp(时间点)
```python
# unix time2datetime
pd.to_datetime(1490195805, unit='s')
# => Timestamp('2017-03-22 15:16:45')

# datetime str2Timestamp
pd.to_datetime("2017-11-01 12:24")
# or setting format
pd.to_datetime("2017年11月1日 12时24分", format='%Y年%m月%d日 %H时%M分')
#=> Timestamp('2017-11-01 12:24:00')
```
- (Attributes)[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html]:  
	- 年月日（year, month, day）
	- 時分秒（hour, minute, second） 

###### DatetimeIndex(时间序列索引)
```python
pd.to_datetime([1490195805.433, 1490195805.433502912], unit='s')
#=>DatetimeIndex(['2017-03-22 15:16:45.433000088', '2017-03-22 15:16:45.433502913'], dtype='datetime64[ns]', freq=None)
```
- (Attributes)[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html]:  
	- 年月日（year, month, day）, 
	- 時分秒（hour, minute, second）
- unix time形式
```python
pd.to_datetime(['2017-03-22 15:16:45.433000088', '2017-03-22 15:16:45.433502913']).astype(int) / 10**9
#=> Float64Index([1490195805.433, 1490195805.433503], dtype='float64')
```
参考: https://stackoverflow.com/questions/54313463/pandas-datetime-to-unix-timestamp-seconds

##### date_range
Return a fixed frequency DatetimeIndex. 参考: [pandas.date_range](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)
###### 常用参数:
- start: Left bound for generating dates.
- end: Right bound for generating dates.
- periods: Number of periods to generate.
- freq: Frequency strings can have multiples. 参考: [timeseries-offset-aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)
  - D: calendar day frequency
  - M: month end frequency
  - Y: year end frequency
  - H: hourly frequency
  - T: minutely frequency
  - S: secondly frequency
  - Q: 季度

###### examples:
```python
pd.date_range(start='1/1/2018', end='1/08/2018')
#=> DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],dtype='datetime64[ns]', freq='D')

# 开始为2018.1.1, 取8个日期，默认间隔为天
pd.date_range(start='1/1/2018', periods=8)
#=> DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],dtype='datetime64[ns]', freq='D')

# 三个月为间隔
pd.date_range(start='1/1/2018', periods=5, freq='3M')
#=> DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31', '2019-01-31'], dtype='datetime64[ns]', freq='3M')

```

##### 日期检索
```python
test_df = pd.DataFrame({'data': range(1, 1000)})
test_df.index = pd.date_range(start='2020-1-1', end='2020-6-1', periods=test_df.shape[0])

# 获取2020年的数据
test_df['2020']
# 获取2020年5月的数据
test_df['2020-5']
# 获取2020年5月1号的数据
test_df['2020-5-1']
# 获取2020年一季度(1,2,3月)的数据
test_df['2020Q1']
# 获取2020年5月1号到2020年5月30号的数据
test_df['2020-5-1':'2020-5-30']

```


#### 聚合类方法 groupby() + agg() 
参考：https://www.cnblogs.com/feffery/p/11468762.html
要进行分组运算第一步当然就是分组，在pandas中对数据框进行分组使用到groupby()方法，其主要使用到的参数为by，这个参数用于传入分组依据的变量名称，**当变量为1个时传入名称字符串即可，当为多个时传入这些变量名称列表**，DataFrame对象通过groupby()之后**返回一个生成器**，需要将其列表化才能得到需要的分组后的子集
```Python
group_df = trajs_with_id_df.groupby(by=['car_id'])[['traj_id', 'traj_points']]
groups = [group for group in group_df]
groups[0]
```
- output:
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200228082637.png)
- 每一个结果都是一个二元组，**元组的第一个元素是对应这个分组结果的分组组合方式，第二个元素是分组出的子集数据框**
```Python
groups = data_df.groupby(by=['assigned_c', 'osmid'])[['x', 'y']].max().reset_index(drop=False)
```
- by=['assigned_c', 'osmid'],分组的组合方式
- [['x', 'y']]分组后取出x和y列进行后续操作
- max() 根据分组对x和y列取最大值
- reset_index(drop=False) 重置索引，便于显示和取值; drop: 是否丢掉原索引
```Python
agg_df = data_df.groupby(by=['assigned_c']).agg({'x': ['mean', 'max', 'min', 'mean', 'std']})
										.reset_index(drop=False)

```
- agg() 即aggregate，聚合；传入字典：操作列为key和相关操作为value

```Python
agg_df = data_df.groupby(by=['assigned_c']).agg(mean_x=pd.NamedAgg(column='x', aggfunc='mean'),
                                        max_x=pd.NamedAgg(column='x', aggfunc='max'),
                                        min_x=pd.NamedAgg(column='x', aggfunc='min'),)
										.reset_index(drop=False)

```
- 使用pd.NamedAgg()来为聚合后的每一列赋予新的名字


### 异常调试

#### 忽略warning全局设置
```Python
import warnings
warnings.filterwarnings('ignore')
```

#### 编码错误（illegal multibyte sequence）
```Python
def get_df(file_path): 
    poi_df = None
    with open(file_path, encoding='gb2312', errors='ignore') as fp:
        poi_df = pd.read_csv(fp, header=0)
        return poi_df
```

#### 显示设置
- 取消科学计数法显示
参考：https://blog.csdn.net/chenpe32cp/article/details/87883420
科学计数法的显示很难阅读。取消科学计数法显示，保留小数后两位。
```python
# 全局设置
pd.set_option('display.float_format',lambda x : '%.2f' % x)
# 或
# 单个DataFrame生效
df = pd.DataFrame(np.random.random(10)**10, columns=['data'])
df.round(3)
```

- 显示所有列或所有行
参考：https://blog.csdn.net/qq_42648305/article/details/89640714
```python
pd.options.display.max_columns = None
pd.options.display.max_rows = None
```


### Geopandas

#### I/O
参考：https://geopandas.org/io.html#writing-spatial-data 
- 支持的格式：
```python
import fiona
fiona.supported_drivers 

{'AeronavFAA': 'r',
 'ARCGEN': 'r',
 'BNA': 'rw',
 'DXF': 'rw',
 'CSV': 'raw',
 'OpenFileGDB': 'r',
 'ESRIJSON': 'r',
 'ESRI Shapefile': 'raw',
 'FlatGeobuf': 'rw',
 'GeoJSON': 'raw',
 'GeoJSONSeq': 'rw',
 'GPKG': 'raw',
 'GML': 'rw',
 'OGR_GMT': 'rw',
 'GPX': 'rw',
 'GPSTrackMaker': 'rw',
 'Idrisi': 'r',
 'MapInfo File': 'raw',
 'DGN': 'raw',
 'OGR_PDS': 'r',
 'S57': 'r',
 'SEGY': 'r',
 'SUA': 'r',
 'TopoJSON': 'r'}

```
- Read Data
  - read_file
  ```python
  geopandas.read_file(fp)
  # gdb or gpkg's layer
  gdb_fp = '/workspace/UrbanFunctionalRegionalization/map_doc/regionalization.gdb'
  thiessen_gdf = gpd.geopandas.read_file(gdb_fp, layer='thiessen_cliped')
  ```
- Write Data
  For a full list of supported formats, type `import fiona; fiona.supported_drivers`
  - Writing to Shapefile
  ```python
  countries_gdf.to_file("countries.shp")
  ```
  - Writing to GeoJSON file
  ```python
  countries_gdf.to_file("countries.geojson", driver='GeoJSON')
  ```
  - Writing to GeoJSON string
  ```python
  countries_gdf.to_json()
  ```
- Writing to GeoPackage
  ```python
  countries_gdf.to_file("package.gpkg", layer='countries', driver="GPKG")
  cities_gdf.to_file("package.gpkg", layer='cities', driver="GPKG")
  ```

#### DataFrame2GeoDataFrame

##### set_geometry
- set_geometry
```python
trajs_gpd_df = trajs_df.set_geometry('line_geo')
```
- 在初始化GeoDataFrame时定义geometry
```python
from shapely.geometry import Point, LineString
geometry = [Point(), Point(), ...] or [LineString([p1, p2, ...]), ...] or ...
gpd.GeoDataFrame(df, geometry=[])
```

##### 空间连接/Spatial Join
参考：https://blog.csdn.net/qq_28360131/article/details/81165168
shaply包的vectorized包含着一些对查询的优化，通过shaply和geopandas一起协作可以达到较好的优化效果。

```python
import shapely.vectorized as sv
point_df = point_df[['id', 'lon', 'lat']]

for row in ploy_gdf.itertuples():
    geometry, area_id = getattr(row, "geometry"), getattr(row, "area_id")
    point_df.loc[sv.contains(geometry, x=point_df.lon, y=point_df.lat), "AreaID"] = area_id

```
- point_df

|      |  id  |   lon   |   lat   |      datetime       | area_id |
| :--: | :--: | :-----: | :-----: | :-----------------: | :-----: |
|  0   |  0   | 116.41  | 39.9084 | 2012-11-02 00:25:03 |   94    |
|  1   |  1   | 116.583 | 40.0793 | 2012-11-02 01:25:41 |   nan   |
|  2   |  2   | 116.34  | 39.9567 | 2012-11-02 02:06:14 |   145   |
|  3   |  3   | 116.343 | 39.9126 | 2012-11-02 02:19:51 |   86    |
|  4   |  4   | 116.334 | 39.906  | 2012-11-02 02:26:55 |   82    |



***

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200307175915.jpg)
