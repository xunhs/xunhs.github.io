---
title: Kaggle-Starter Simple Framework For Prediction Issues
date: 2020-02-25T09:16:50+00:00
categories:
  - 收藏
tags:
  - Kaggle
  - 预测模型
  - 比赛
---


> kaggle中处理预测问题通用流程/基本框架，参考
[How to get to TOP 25% with Simple Model (sklearn)](https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn)

case complete: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

<!--more-->



目录

[TOC]



### Init

Adding needed libraries and reading data

```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

```


### Checking for NAs

Checking for missing data
```Python
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]

```

- output  

|              |   Train |   Test |
|:-------------|--------:|-------:|
| Alley        |    1369 |   1352 |
| BsmtCond     |      37 |     45 |
| BsmtExposure |      38 |     44 |
| BsmtFinSF1   |       0 |      1 |
| BsmtFinSF2   |       0 |      1 |
| BsmtFinType1 |      37 |     42 |
| BsmtFinType2 |      38 |     42 |
| BsmtFullBath |       0 |      2 |
| BsmtHalfBath |       0 |      2 |
| BsmtQual     |      37 |     44 |
| BsmtUnfSF    |       0 |      1 |
| Electrical   |       1 |      0 |
| Exterior1st  |       0 |      1 |
| Exterior2nd  |       0 |      1 |
| Fence        |    1179 |   1169 |
| FireplaceQu  |     690 |    730 |
| Functional   |       0 |      2 |
| GarageArea   |       0 |      1 |
| GarageCars   |       0 |      1 |
| GarageCond   |      81 |     78 |
| GarageFinish |      81 |     78 |
| GarageQual   |      81 |     78 |
| GarageType   |      81 |     76 |
| GarageYrBlt  |      81 |     78 |
| KitchenQual  |       0 |      1 |
| LotFrontage  |     259 |    227 |
| MSZoning     |       0 |      4 |
| MasVnrArea   |       8 |     15 |
| MasVnrType   |       8 |     16 |
| MiscFeature  |    1406 |   1408 |
| PoolQC       |    1453 |   1456 |
| SaleType     |       0 |      1 |
| TotalBsmtSF  |       0 |      1 |
| Utilities    |       0 |      2 |




### Importing public functions

```Python
# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing estimator
    print(estimator)
    # Printing train scores
    print("Train")
    get_score(prediction_train, y_trn)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)
```

### Splitting to features and labels and deleting variables I don't need


```Python
# Spliting to features and lables and deleting variable I don't need
train_labels = train.pop('SalePrice')

features = pd.concat([train, test], keys=['train', 'test'])

# I decided to get rid of features that have more than half of missing information or do not correlate to SalePrice
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)

```

- `features = pd.concat([train, test], keys=['train', 'test'])` 为上下拼接，并添加'train', 'test'标签

- 有人说**不建议在一开始丢弃变量**
> Don't discard variables unless you have a good reason for it. Note that it is not a good reason to say it's not correlated. **Tree algorithms can use the information and are not harmed by including it**. A good reason for **excluting could be using KNN**. The other main reason for discarding variables are if they are correlated with being in the test or training set e.g. in many competitions you goal is to predict out of time. So including time can be very harmfull. 


### Filling NAs and converting features

- 取代NA值得方法
	- 数值型
		- 众数（filling with most popular values）
			```Python
			features['MSZoning'].fillna(features['MSZoning'].mode()[0])
			```
		- 0（filling with 0）
			```Python
			features['TotalBsmtSF'].fillna(0)
			```
		- 平均值（filling with means）
			```Python
			features['LotFrontage'].fillna(features['LotFrontage'].mean())
			```
	- category型
		- NA用新值代替
			```Python
			features['Alley'].fillna('NOACCESS')
			```

```Python
# MSSubClass as str
features['MSSubClass'] = features['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())

# Alley  NA in all. NA means no access
features['Alley'] = features['Alley'].fillna('NOACCESS')

# Converting OverallCond to str
features.OverallCond = features.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

# TotalBsmtSF  NA in pred. I suppose NA means 0
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# KitchenAbvGr to categorical
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str) features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
features['GarageCars'] = features['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

```

- 根据变量描述和特征逐一分析 [data_description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) 
- MSSubClass: Identifies the type of dwelling involved in the sale. 根据释义， MSSubClass更应该是一种category类型，因此代码中通过类型变换为字符串再通过后续变换从而转化成 category 类型


### Log transformation
- log变换


```Python
# Our SalesPrice is skewed right (check plot below). I'm logtransforming it. 
ax = sns.distplot(train_labels)
## Log transformation of labels
train_labels = np.log(train_labels)
## Now it looks much better
ax = sns.distplot(train_labels)

```
- output
	![3Y1GDI.png](https://s2.ax1x.com/2020/02/25/3Y1GDI.png)
	After transform:
	![3Y125T.png](https://s2.ax1x.com/2020/02/25/3Y125T.png)

- 注意**最后的预测值要用 np.exp() 变换为原维度**


### Standardizing numeric data
- 数值类型变量标准化

```Python
## Standardizing numeric features
numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
ax = sns.pairplot(numeric_features_standardized)

```
- output

![](https://s2.ax1x.com/2020/02/25/3Y3SsA.md.png)



### Converting categorical data to dummies


```Python
# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)

# Getting Dummies from all other categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)

```
- Condition 和 Exterior 为特殊处理


### Obtaining standardized dataset

```Python
### Copying features
features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized)

```

### Splitting train and test features
```Python
### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

```
- 两类特征数据，一类未标准化 一类标准化过的 适应不同模型
	- tree-based model无需标准化
	- 线性模型需标准化


### Splitting to train and validation sets
```Python
### Shuffling train sets
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)

### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)
```

- shuffle 打乱数据顺序

### First level models
#### Elastic Net
using ElasticNetCV estimator to choose best alpha and l1_ratio for my Elastic Net model

```Python
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], 
                                    l1_ratio=[.01, .1, .5, .9, .99], 
                                    max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)

```

- output

```python
Train
R2: 0.9015542514742068
RMSE: 0.11906917427161198
Test
R2: 0.8983659574062
RMSE: 0.10972333168339993
```

```Python

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```
- output

`Accuracy: 0.88 (+/- 0.09)`



#### Gradient Boosting

We use a **lot of features** and have many **outliers**. So I'm using **`max_features='sqrt'` to reduce overfitting of my model**. I also use **`loss='huber'` because it more tolerant to outliers**. **All other hyper-parameters was chosen using GridSearchCV**.

```Python
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, 
                                           learning_rate=0.05, 
                                           max_depth=3, 
                                           max_features='sqrt',
                                           min_samples_leaf=15, 
                                           min_samples_split=10, 
                                           loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

```

```python
Train
R2: 0.9618766649349884
RMSE: 0.07600479015539144
Test
R2: 0.9046920143397521
RMSE: 0.10696670219256725
```


```Python
# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

```
`Accuracy: 0.89 (+/- 0.04)`


#### Ensembling final model
final ensemble model is **an average of Gradient Boosting and Elastic Net predictions**. But before that I retrained my models on all train data.
 

```Python
# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

```
- 融合模型，取输出的平均值

### Saving to CSV
```Python
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('2020-02-25.csv', index =False)   

```

***

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200227170525.jpg)