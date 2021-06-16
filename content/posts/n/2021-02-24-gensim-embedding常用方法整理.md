---
title: POI-type Embedding构建流程札记
date: 2021-02-24T01:44:11+00:00
categories:
  - 收藏
tags:
  - Embedding
  - Gensim
  - POIs
  - Python
  - 城市功能区
  - 札记

---
> 本文对POI-type Embedding的构建流程进行整理。


<!--more-->

### POI数据准备
- 高德地图兴趣点2018-POI（Point of Interest）数据，数据共享地址：https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/WSXCNM
- 数据入库（这里使用的是ES，方便快速查询及可视化）（参考之前写的ES操作的文章）


### 构建语料库
- 生成随机点
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210224095729.png)
- 构建随机点语料库
```python
# Building K-NN Corpus
# 根据随机点以及指定距离，寻找附近POI点，构建“语料库”
# 此处POI type选择的是高德地图的type3
# %%
import geopandas as gpd
from tqdm import tqdm
import os, joblib
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
client = Elasticsearch(hosts="192.168.123.87:9200")
# %%
random_points_fp = os.path.join('.', 'RandomPoints.shp')
random_points_gdf = gpd.read_file(random_points_fp)
random_points_gdf.head()
# %%
def get_neighbors(client, index, geo_point, distance, geo_unit="m"):
    s = Search(using=client, index=index)
    s = s.filter(
        "geo_distance", distance=f"{distance}{geo_unit}", wgs_location=geo_point
    )
    _sort_json = {
        "_geo_distance": {
            "wgs_location": geo_point,
            "order": "asc",
            "unit": geo_unit,
            "mode": "min",
            "distance_type": "arc",
        }
    }
    s = s.sort(_sort_json)
    total = s.count()
    s = s[0:total]
    response = s.execute()
    return total, response
# %%
index = "北京市"
distance = 500
geo_unit = "m"
corpus_dict = {}
for (point_id, lat, lon) in tqdm(zip(random_points_gdf['PointID'], random_points_gdf['lat'], random_points_gdf['lon'])):
    geo_point = {"lat": lat, "lon": lon}
    total, neighbors = get_neighbors(client, index, geo_point, distance, geo_unit)
    if total > 10: # 至少10个点
        corpus_dict[point_id] = []
        for p in neighbors.hits.hits:
            distance = p.sort[0]
            corpus_dict[point_id].append(p._source.type_3)
# %%
# 保存变量
corpus_fp = os.path.join('.', 'EmbeddingCorpus.dat')
joblib.dump(value=corpus_dict, filename=corpus_fp)
# # 载入变量
# corpus_fp = os.path.join('.', 'EmbeddingCorpus.dat')
# corpus_dict = joblib.load(corpus_fp)
```


### 使用gensim进行POI-type embedding的训练
```python
import os, joblib
from gensim.models.word2vec import Word2Vec
import logging
import json


# 载入变量
corpus_fp = os.path.join('.', 'EmbeddingCorpus.dat')
corpus_dict = joblib.load(corpus_fp)
_corpora = corpus_dict.values()


'''
语料库模板如下
_corpora= [
                [type1, type2, ...,],
                [],
                ...
            ]

'''

# %%

#获取日志信息
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def func(corpora, path_name, model_size=100):

    model = Word2Vec(corpora,
                    size=model_size, # 特征向量的维度
                    min_count=1,  # 需要计算词向量的最小词频
                    window=5, # 词向量上下文最大距离
					sample = 1e-3, # 高频词汇的随机降采样的配置阈值
                    sg=1, # 0: CBOW; 1: Skip-Gram
					hs = 1,  #为 1 用hierarchical softmax   0 negative sampling
                    iter=100, # 随机梯度下降法中迭代的最大次数
					workers=8 # 开启线程个数
                    )

    model.save(r'Embedding{}.word2vec'.format(path_name))

    # 保存向量
    words = []
    for c in corpora:
        words.extend(c)
    words = list(set(words))
    print('总共{}个单词'.format(len(words)))

    with open(r'Embedding{}.json'.format(path_name), 'w+', encoding='utf8') as fp:
        save_json = {}
        for w in words:
            save_json[w] = [float(_) for _ in model.wv[w]]
        json.dump(save_json, fp)

model_size = 128  # 词向量的维度
func(_corpora, 'POI', model_size)
```

### 向量查询检索
```python
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

# 载入预训练
# 方式一：载入已训练模型
w2v_model = Word2Vec.load('Embedding128.word2vec')
# 方式二：载入已训练向量（如OpenNE导出的向量）
with open('/workspace/UrbanFunctionalRegionalization/result/openne/deepwalk_sz_ep10.txt', 'r') as fp:
    w2v_wv = KeyedVectors.load_word2vec_format(fp, binary=False)
# w2v_wv为KeyedVectors格式，即等同于w2v_model.wv

```

- 获取单词列表：`word_list = w2v_model.wv.index2word` 或 `word_list = w2v_wv.index2word`
- 获取单词向量：`w2v_model.wv.get_vector('机场相关')` `w2v_wv.get_vector('机场相关')`
- 查询两个词的相似度：
```python
s_word_1 = "机场相关"
s_word_2 = "脑科医院"
f_word_sim = w2v_model.similarity(s_word_1, s_word_2)
```
- 查询单词最相似的单词（默认前10）：`w2v_model.most_similar("脑科医院")`


### POI-type Embedding案例：应用于城市功能区分类
- POI数据与区块数据（如交通分析小区TAZ，空间格网等）进行空间关联（spatial join）
	- POI空间关联结果：`POIJoinTaz.shp`
	- 交通分析小区（with functions）：`eulucJoinTaz.shp`
- 计算doc embedding，这里使用简单使用的mean embedding
```python
from gensim.models.word2vec import Word2Vec
import geopandas as gpd 
import pandas as pd 
import numpy as np 
import os 
w2v_model = Word2Vec.load('Embedding128.word2vec')
root_dir = os.path.join('..', 'OriginalPublicData')
POI_gdf = gpd.read_file(os.path.join(root_dir, 'POIJoinTaz.shp'))
taz_gdf = gpd.read_file(os.path.join(root_dir, 'eulucJoinTaz.shp'))
# %%
taz_group = POI_gdf.groupby(by=['TAZ_FID'])
# 计算mean embedding
mean_feature_dict = {}
vector_list = []
for (taz_FID, taz_sub_gdf) in taz_group:
    type_list = taz_sub_gdf['type_3']
    for t in type_list:
        try:
            v = w2v_model.wv.get_vector(t)
        except:
            pass
        vector_list.append(v)
    mean_feature_dict[taz_FID] = np.mean(np.array(vector_list), axis=0)
# %%
# mean_feature_dict 转为 DataFrame 保存
_dict = {}
_dict['taz_fid'] = []
for i in range(128):
    feature_name = f'feature_{i}'
    _dict[feature_name] = []
for _fid, mean_feature in mean_feature_dict.items():
    _dict['taz_fid'].append(_fid)
    for i, f in enumerate(list(mean_feature)):
        feature_name = f'feature_{i}'
        _dict[feature_name].append(f)
mean_feature_df = pd.DataFrame(_dict)
mean_feature_path = os.path.join('.', 'TAZDocEmbedding.txt')
mean_feature_df.to_csv(mean_feature_path, header=True, index=False)
```
- 构建分类器
```python
# 分类
import geopandas as gpd
import pandas as pd
import os
from  sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# %%
taz_doc_embedding_path = os.path.join('.', 'TAZDocEmbedding.txt')
taz_doc_embedding_df = pd.read_csv(taz_doc_embedding_path, header=0)
root_dir = os.path.join('..', 'OriginalPublicData')
taz_gdf = gpd.read_file(os.path.join(root_dir, 'eulucJoinTaz.shp'))
classifier_df = pd.merge(taz_doc_embedding_df,
        taz_gdf[['TAZ_FID', 'Level1', 'Level2']],
        left_on= 'taz_fid',
        right_on='TAZ_FID',
        how='inner')
# %%
# 构建训练数据集
# 样本不平衡问题，增加小样本的样本量
sample_1 = classifier_df.query('Level1 == 1').sample(n=500, random_state=2021)
sample_2 = classifier_df.query('Level1 == 5').sample(n=500, random_state=2021)
sample_3 = classifier_df.query('Level1 == 2')
sample_4 = classifier_df.query('Level1 == 3')
concat_list = [sample_1, sample_2] + [sample_3] * 2 + [sample_4] * 4
classifier_df = pd.concat(concat_list, axis=0)
feature_names = list(taz_doc_embedding_df.columns)[1:]
x = classifier_df[feature_names]
scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)
y = classifier_df['Level1']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2021)
# %%
# 模型选择
models = [
    ('LR'   , LogisticRegression()),
    # ('LDA'  , LinearDiscriminantAnalysis()),
    # ('KNN'  , KNeighborsClassifier()),
    # ('CART' , DecisionTreeClassifier()),
    # ('NB'   , GaussianNB()),
    ('SVM'  , SVC(probability=True)),
    # ('AB'   , AdaBoostClassifier()),
    # ('GBM'  , GradientBoostingClassifier()),
    ('RF'   , RandomForestClassifier()),
    # ('ET'   , ExtraTreesClassifier())
]
def run_models(x, y, models):
    num_folds = 10
    scoring = 'accuracy'
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=2021)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    return names, results
names, results = run_models(x, y, models)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Results:
LR: 0.356372 (0.070240)
SVM: 0.333471 (0.061519)
RF: 0.649130 (0.022192)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# %%
# 分类报告
model = RandomForestClassifier()
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 获取混淆矩阵
m = confusion_matrix(test_y, pred_test_y)
print('混淆矩阵：', m, sep='\n')
# 获取分类报告
r = classification_report(test_y, pred_test_y)
print('分类报告：', r, sep='\n')
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Results:
混淆矩阵：
[[59 24 12 52]
 [12 97  0 14]
 [ 0  0 66  0]
 [49 28 13 59]]
分类报告：
              precision    recall  f1-score   support

           1       0.49      0.40      0.44       147
           2       0.65      0.79      0.71       123
           3       0.73      1.00      0.84        66
           5       0.47      0.40      0.43       149

    accuracy                           0.58       485
   macro avg       0.58      0.65      0.61       485
weighted avg       0.56      0.58      0.56       485
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# %%
# 尝试模型融合 ensemble.stacking
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('SVM'  , SVC(probability=True))
]
model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
# 获取混淆矩阵
m = confusion_matrix(test_y, pred_test_y)
print('混淆矩阵：', m, sep='\n')
# 获取分类报告
r = classification_report(test_y, pred_test_y)
print('分类报告：', r, sep='\n')
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Results:
混淆矩阵：
[[61 22  8 56]
 [16 93  0 14]
 [ 0  0 64  2]
 [61 22  9 57]]
分类报告：
              precision    recall  f1-score   support

           1       0.44      0.41      0.43       147
           2       0.68      0.76      0.72       123
           3       0.79      0.97      0.87        66
           5       0.44      0.38      0.41       149

    accuracy                           0.57       485
   macro avg       0.59      0.63      0.61       485
weighted avg       0.55      0.57      0.56       485
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

------------

![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210224094404.jpg)





