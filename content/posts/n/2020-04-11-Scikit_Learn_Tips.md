---
layout: post
cid: 208
title: Scikit Learn Tips
slug: 209
date: 2020-04-11T09:21:12+00:00
status: publish
author: Ethan
toc: true
categories:
  - 收藏
tags:
  - Python
  - sklearn

---


> Refer to [scikit-learn-tips](https://github.com/justmarkham/scikit-learn-tips), [Scikit-learn 0.22新版本发布](https://cloud.tencent.com/developer/article/1555141)。
整理一些自己感兴趣，经常用到的

<!--more-->


### 特征工程


#### 归一化/标准化/正则化
参考: [cnblogs](https://www.cnblogs.com/chaosimple/p/4153167.html)
##### StandardScaler
Z-Score，或者去除均值和方差缩放
```python
from  sklearn import preprocessing
import numpy as np
x = np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])

# 使用sklearn.preprocessing.StandardScaler类，
# 使用该类的好处在于可以保存训练集中的参数（均值、方差）
# 直接使用其对象转换测试集数据。
scaler = preprocessing.StandardScaler().fit(x)
scaler.mean_
scaler.std_
scaler.transform(x)  #跟上面的结果是一样的
```

##### MinMaxScaler
将属性缩放到一个指定范围,也是就是(x-min)/(max-min)
```python
x_train = np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])
 
min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
print(x_train_minmax)
# 当然，在构造类对象的时候也可以直接指定最大最小值的范围：
# feature_range = (min, max)，此时应用的公式变为：
# x_std = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
# x_scaled = X_std/(max-min)+min
```

##### Normalization
```python
# 可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换
normalizer = preprocessing.Normalizer().fit(x)
print(normalizer)
normalizer.transform(x)
```


#### ColumnTransformer-make_column_transformer/行处理

Use `ColumnTransformer` to apply different preprocessing to different columns:

- select from DataFrame columns by name
- passthrough(保留) or drop(丢掉) unspecified columns
- 引申：sklearn.preprocessing(TODO): [官网](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)，[cnblogs](https://www.cnblogs.com/chaosimple/p/4153167.html)
- sklearn.impute.SimpleImputer: 填补缺失值, 参考[官网](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html), [zhihu](https://zhuanlan.zhihu.com/p/83173703)

```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)

cols = ['Fare', 'Embarked', 'Sex', 'Age']
X = df[cols]

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
imp = SimpleImputer()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),  # apply OneHotEncoder to Embarked and Sex
    (imp, ['Age']),              # apply SimpleImputer to Age
    remainder='passthrough')     # include remaining column (Fare) in the output
    
# column order: Embarked (3 columns), Sex (2 columns), Age (1 column), Fare (1 column)
ct.fit_transform(X)

'''
output:
array([[ 0.    ,  0.    ,  1.    ,  0.    ,  1.    , 22.    ,  7.25  ],
       [ 1.    ,  0.    ,  0.    ,  1.    ,  0.    , 38.    , 71.2833],
       [ 0.    ,  0.    ,  1.    ,  1.    ,  0.    , 26.    ,  7.925 ],
       [ 0.    ,  0.    ,  1.    ,  1.    ,  0.    , 35.    , 53.1   ],
       [ 0.    ,  0.    ,  1.    ,  0.    ,  1.    , 35.    ,  8.05  ],
       [ 0.    ,  1.    ,  0.    ,  0.    ,  1.    , 31.2   ,  8.4583]])
       
'''
```


#### ColumnTransformer-行选择
There are SEVEN ways to select columns using ColumnTransformer:

1. column name
2. integer position
3. slice
4. boolean mask
5. regex pattern
6. dtypes to include
7. dtypes to exclude

```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)
cols = ['Fare', 'Embarked', 'Sex', 'Age']
X = df[cols]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer  # new in 0.20
from sklearn.compose import make_column_selector     # new in 0.22

# all SEVEN of these produce the same results
ct = make_column_transformer((ohe, ['Embarked', 'Sex']))
ct = make_column_transformer((ohe, [1, 2]))
ct = make_column_transformer((ohe, slice(1, 3)))
ct = make_column_transformer((ohe, [False, True, True, False]))
ct = make_column_transformer((ohe, make_column_selector(pattern='E|S')))
ct = make_column_transformer((ohe, make_column_selector(dtype_include=object)))
ct = make_column_transformer((ohe, make_column_selector(dtype_exclude='number')))

ct.fit_transform(X)
```


#### pipeline
Chains together multiple steps: output of each step is used as input to the next step. Makes it easy to apply the same preprocessing to train and test!
```python
import pandas as pd
import numpy as np
train = pd.DataFrame({'feat1':[10, 20, np.nan, 2], 'feat2':[25., 20, 5, 3], 'label':['A', 'A', 'B', 'B']})
test = pd.DataFrame({'feat1':[30., 5, 15], 'feat2':[12, 10, np.nan]})

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

imputer = SimpleImputer()
clf = LogisticRegression()

# 2-step pipeline: impute missing values, then pass the results to the classifier
pipe = make_pipeline(imputer, clf)

features = ['feat1', 'feat2']
X, y = train[features], train['label']
X_new = test[features]

# pipeline applies the imputer to X before fitting the classifier
pipe.fit(X, y)

# pipeline applies the imputer to X_new before making predictions
# note: pipeline uses imputation values learned during the "fit" step
pipe.predict(X_new)
```

#### 缺失值处理
##### 标记缺失数值并将此标记作为新的特征
Add a missing indicator to encode "missingness" as a feature(在处理缺失数据的时候，标记缺失数值并将此标记作为新的特征)
When imputing missing values, you can preserve info about which values were missing and use THAT as a feature!Why? Sometimes there's a relationship between "missingness" and the target/label you are trying to predict.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

X = pd.DataFrame({'Age':[20, 30, 10, np.nan, 10]})
# impute the mean
imputer = SimpleImputer()
imputer.fit_transform(X)

'''
output:
array([[20. ,  0. ],
       [30. ,  0. ],
       [10. ,  0. ],
       [17.5,  1. ],
       [10. ,  0. ]])
'''
```

##### KNNImputer / IterativeImpute:
Need something better than SimpleImputer for missing value imputation?
Try KNNImputer or IterativeImputer (inspired by R's MICE package). Both are multivariate approaches (they take other features into account!)
另参考：[csdn](https://blog.csdn.net/jingyi130705008/article/details/82796283)

```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)
cols = ['SibSp', 'Fare', 'Age']
X = df[cols]

# new in 0.21, and still "experimental" so it must be enabled explicitly
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

impute_it = IterativeImputer()
impute_it.fit_transform(X)
'''
output:
array([[ 1.        ,  7.25      , 22.        ],
       [ 1.        , 71.2833    , 38.        ],
       [ 0.        ,  7.925     , 26.        ],
       [ 1.        , 53.1       , 35.        ],
       [ 0.        ,  8.05      , 35.        ],
       [ 0.        ,  8.4583    , 28.50639495]])
'''

# new in 0.22
from sklearn.impute import KNNImputer
impute_knn = KNNImputer(n_neighbors=2)
impute_knn.fit_transform(X)


'''
output:
array([[ 1.    ,  7.25  , 22.    ],
       [ 1.    , 71.2833, 38.    ],
       [ 0.    ,  7.925 , 26.    ],
       [ 1.    , 53.1   , 35.    ],
       [ 0.    ,  8.05  , 35.    ],
       [ 0.    ,  8.4583, 30.5   ]])
'''

```



### 训练


#### random_state:
Set a "random_state" to make your code reproducible
Ensures that a "random" process will output the same results every time, which makes your code reproducible (by you and others!)

```python
from sklearn.model_selection import train_test_split
# any positive integer can be used for the random_state value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

```
#### 划分三类：train, test, val
```python

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

print(x_train, x_val, x_test)
```


#### cross-validate and grid search (交叉验证，网格搜索):
You can cross-validate and grid search an entire pipeline!

Preprocessing steps will automatically occur AFTER each cross-validation split, which is critical if you want meaningful scores.

```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain')

cols = ['Sex', 'Name']
X = df[cols]
y = df['Survived']

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear', random_state=1)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(ct, clf)

# Cross-validate the entire pipeline (not just the model)
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
'''
output:
0.8024543343167408
'''

# Find optimal tuning parameters for the entire pipeline
# specify parameter values to search
params = {}
params['columntransformer__countvectorizer__min_df'] = [1, 2]
params['logisticregression__C'] = [0.1, 1, 10]
params['logisticregression__penalty'] = ['l1', 'l2']
# try all possible combinations of those parameter values
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

# what was the best score found during the search?
grid.best_score_

# which combination of parameters produced the best score?
grid.best_params_

```


#### RandomizedSearchCV (随机化网格搜索)
GridSearchCV taking too long? Try **RandomizedSearchCV** with a small number of iterations.
Make sure to specify a distribution (instead of a list of values) for continuous parameters!

```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain')
X = df['Name']
y = df['Survived']


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(CountVectorizer(), MultinomialNB())

# cross-validate the pipeline using default parameters
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# specify parameter values to search (use a distribution for any continuous parameters)
import scipy as sp
params = {}
params['countvectorizer__min_df'] = [1, 2, 3, 4]
params['countvectorizer__lowercase'] = [True, False]
params['multinomialnb__alpha'] = sp.stats.uniform(scale=1)

# try "n_iter" random combinations of those parameter values
from sklearn.model_selection import RandomizedSearchCV
rand = RandomizedSearchCV(pipe, params, n_iter=10, cv=5, scoring='accuracy', random_state=1)
rand.fit(X, y);

# what was the best score found during the search?
rand.best_score_

# which combination of parameters produced the best score?
rand.best_params_

```

#### 网格化搜索结果输出
Hyperparameter search results (from GridSearchCV or RandomizedSearchCV) can be converted into a pandas DataFrame.
Makes it far easier to explore the results!

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

# convert results into a DataFrame
results = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
# sort by test score
results.sort_values('rank_test_score')
```
***

#### fit & transform
the difference between "fit" and "transform"  

- "fit": transformer learns something about the data
- "transform": it uses what it learned to do the data transformation




### 模型:
#### 套用简单模型。尤其对于高维稀疏数据的regression问题
```python
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

models = [
    ('LR'   , LogisticRegression()),
    ('LDA'  , LinearDiscriminantAnalysis()),
    ('KNN'  , KNeighborsClassifier()),
    ('CART' , DecisionTreeClassifier()),
    ('NB'   , GaussianNB()),
    ('SVM'  , SVC(probability=True)),
    ('AB'   , AdaBoostClassifier()),
    ('GBM'  , GradientBoostingClassifier()),
    ('RF'   , RandomForestClassifier()),
    ('ET'   , ExtraTreesClassifier())
]

def run_models(x, y, models):
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=123)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results

names, results = run_models(X, Y, models)
"""
得到的结果:
LR: 0.803470 (0.009425)
LDA: 0.797354 (0.011074)
KNN: 0.772755 (0.013865)
CART: 0.719289 (0.018143)
NB: 0.694681 (0.019061)
SVM: 0.798207 (0.010633)
AB: 0.805602 (0.010468)
GBM: 0.804893 (0.013025)
RF: 0.781855 (0.010472)
ET: 0.769192 (0.016778)
"""
```

#### 模型融合(ensemble)

参考：[A Complete ML Pipeline Tutorial (ACU ~ 86%) | Kaggle](https://www.kaggle.com/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86)

##### voting 
```python
from sklearn.ensemble import VotingClassifier
"""
Ensemble from the best models. 
Basic Voting.
"""
param = {'C': 0.01, 'penalty': 'l2'}
model1 = LogisticRegression(**param)

param = {'learning_rate': 0.1, 'n_estimators': 170}
model2 = AdaBoostClassifier(**param)

param = {'learning_rate': 0.1, 'n_estimators': 70}
model3 = GradientBoostingClassifier(**param)

estimators = [('LR', model1), ('AB', model2), ('GB', model3)]

kfold = StratifiedKFold(n_splits=10, random_state=123)
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold, scoring='accuracy')
results.mean()
```

##### stacking
###### StackingClassifier 和 StackingRegressor 

```python
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
clf.fit(X_train, y_train).score(X_test, y_test)
```

###### mlens
```python
"""
Stacking Model using lib: mlens.
"""

def get_models():
    """Generate a library of base learners."""
    param = {'C': 0.01, 'penalty': 'l2'}
    model1 = LogisticRegression(**param)

    param = {'learning_rate': 0.1, 'n_estimators': 170}
    model2 = AdaBoostClassifier(**param)

    param = {'learning_rate': 0.1, 'n_estimators': 70}
    model3 = GradientBoostingClassifier(**param)

    param = {'n_neighbors': 23}
    model4 = KNeighborsClassifier(**param)

    param = {'C': 1.7, 'kernel': 'rbf', 'probability':True}
    model5 = SVC(**param)

    param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    model6 = DecisionTreeClassifier(**param)

    model7 = GaussianNB()

    model8 = RandomForestClassifier()

    model9 = ExtraTreesClassifier()

    models = {'LR':model1, 'ADA':model2, 'GB':model3,
              'KNN':model4, 'SVM':model5, 'DT':model6,
              'NB':model7, 'RF':model8,  'ET':model9
              }

    return models

base_learners = get_models()
meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    loss="exponential",
    max_features=6,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.001, 
    random_state=123
)

from mlens.ensemble import SuperLearner

# Instantiate the ensemble with 10 folds
sl = SuperLearner(
    folds=10,
    random_state=123,
    verbose=2,
    backend="multiprocessing"
)

# Add the base learners and the meta learner
sl.add(list(base_learners.values()), proba=True) 
sl.add_meta(meta_learner, proba=True)

# Train the ensemble
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size=0.2, random_state=0)

sl.fit(X_train, Y_train)

# Predict the test set
p_sl = sl.predict_proba(X_test)

pp = []
for p in p_sl[:, 1]:
    if p>0.5:
        pp.append(1.)
    else:
        pp.append(0.)

print("\nSuper Learner Accuracy score: %.8f" % (Y_test== pp).mean())
```


### 结果评定/验证
#### 特征的重要性
sklearn.inspection.permutation_importance， 可以用来估计每个特征的重要性
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

X, y = make_classification(random_state=0, n_features=5, n_informative=3)
rf = RandomForestClassifier(random_state=0).fit(X, y)
result = permutation_importance(rf, X, y, n_repeats=10, random_state=0,
                                n_jobs=-1)

fig, ax = plt.subplots()
sorted_idx = result.importances_mean.argsort()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=range(X.shape[1]))
ax.set_title("Permutation Importance of each feature")
ax.set_ylabel("Features")
fig.tight_layout()
plt.show()
```

#### ROC曲线
Easily compare multiple ROC curves in a single plot
```python
import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain', header=0)
cols = ['Pclass', 'Fare', 'SibSp']
X = df[cols]
y = df['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr.fit(X_train, y_train);
dt.fit(X_train, y_train);
rf.fit(X_train, y_train);

disp = plot_roc_curve(lr, X_test, y_test)
plot_roc_curve(dt, X_test, y_test, ax=disp.ax_);
plot_roc_curve(rf, X_test, y_test, ax=disp.ax_);

```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200424210215.png)

***
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200415214145.jpg)