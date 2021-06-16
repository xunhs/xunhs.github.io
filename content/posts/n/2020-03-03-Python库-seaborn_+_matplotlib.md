---
title: Python库-seaborn + matplotlib
date: 2020-03-03T09:18:18+00:00
author: Ethan
categories:
  - 收藏
tags:
  - seaborn
  - Python
  - 可视化
  - 学习笔记
  - matplotlib
---


> seaborn是python里面做数据分析和机器学习常用的可视化库。它对matplotlib进行了深度封装，从而可以用非常简单的api接口绘制相对复杂的图形，提供对数据的深入认识。

<!--more-->



### VIsual Vocabulary

![Visual-vocabulary-chinese-simplified](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/Visual-vocabulary-chinese-simplified.wb65xleas8g.jpg)



### 引用
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

```
### sns.set
```python
sns.set(style='white',palette='muted',color_codes=True)
```
- style为图表的背景主题，有5种主题可以选择：
  darkgrid 黑色网格（默认）
  whitegrid 白色网格
  dark 黑色背景
  white 白色背景
  ticks 四周都有刻度线的白背景
- palette为设置主体颜色，有6种可以选择：
  deep,muted,pastel,bright,dark,colorblind

### matplotlib.pyplot

#### cheatsheets
![cheatsheets-1](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153239.png)
![cheatsheets-2](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153314.png)

#### handout
![handout-beginner](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153415.png)
![handout-intermediate](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153437.png)
![handout-tips](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153448.png)	

#### 创建画布与创建子图
```python
fig = plt.figure(figsize=(8,6),dpi=300)  #设置画布大小及分辨率
ax = fig.add_subplot(2,1,1)  #创建一个2行1列的子图，绘制第1张子图
```
matplotlib.pyplot[元素结构图](https://zhuanlan.zhihu.com/p/93423829)
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303165149.jpg)

图像的各个部位名称
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303165506.jpg)

#### 细节处理
```python
ax.set_title('Title',fontsize=18) # 设置标题
ax.set_xlabel('xlabel', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic') # x轴label
ax.set_ylabel('ylabel', fontsize='x-large',fontstyle='oblique') # y轴label
ax.legend() # 

ax.set_aspect('equal') 
ax.minorticks_on() 
ax.set_xlim(0,16)  # x轴范围
ax.grid(which='minor', axis='both') # 背景网格

ax.xaxis.set_tick_params(rotation=45,labelsize=18,colors='w') 
start, end = ax.get_xlim() 
ax.xaxis.set_ticks(np.arange(start, end,1)) 
ax.yaxis.tick_right()
```

#### 保存绘图
```python
import matplotlib.pyplot as plt
import os 
eval_root = '/workspace/UrbanFunctionalRegionalization/result/evaluation'

#分辨率
plt.rcParams['figure.dpi'] = 300 
fig, ax = plt.subplots(figsize=(12, 7))
ax = sns.pointplot(x="threshold", y="purity", data=eval_metrics_df)

ax.set_xlabel('min region size', fontsize=14,)
ax.set_ylabel('purity', fontsize=14,)
# ax.set_xlim(0,100) 
# ax.set_ylim(0,1)


fig_fp = os.path.join(eval_root, 'purity.jpg')
fig.savefig(fig_fp, dpi=300)
```


### seaborn cheatsheets
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/autumn/20200713153816.png)

### Categorical plots

#### [countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html)
通过countplot会绘制出，每个值在样本中出现的次数。
- Show value counts for a **single categorical variable**  

```python
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
ax = sns.countplot(x="day", data=tips)
```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303111508.png)

- Show value counts for **two categorical variables**  
```python
ax= sns.countplot(x='day', hue='smoker', data=tips)
# ax= sns.countplot(x='day', hue='smoker', data=tips) # 横向显示
```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303111848.png)

#### [barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot)
barplot主要用来描述样本的**均值和置信区间**（置信区间本质上应该算是对整个分布的预估，而不仅仅是展示当前样本里面的信息）
- Draw a set of vertical bar plots grouped by a categorical variable  

```python
ax = sns.barplot(x="day", y="total_bill", data=tips)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303115201.png)
- barplot是一个柱状图上面加一个黑线。柱状图的值默认情况下对应的要显示的样本的均值，而黑线默认情况则标识了95%的置信区间。
何为95%的置信区间？95%的置信区间指的是对于当前样本所属的分布而言，当有个新的值产生时，这个值有95%的可能性在该区间内，5%的可能性不在该区间内。

- 显示标准差而非置信区间
```python
ax = sns.barplot(x="day", y="total_bill", data=tips, order=['Sun', 'Sat', 'Fri', 'Thur'], ci='sd', capsize=.1)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303120525.png)

	- 参数说明:
		- order: 控制x轴的显示顺序
		- ci: (float or “sd” or None, optional).Size of confidence intervals to draw around estimated values. If “sd”, skip bootstrapping and draw the standard deviation of the observations. If None, no bootstrapping will be performed, and error bars will not be drawn.
		- capsize: 竖线的上下添加一个“小勾勾”

#### [pointplot](https://seaborn.pydata.org/generated/seaborn.pointplot.html#seaborn.pointplot)
含义与barplox相同，表现形式变更
```python
ax = sns.pointplot(x="day", y="total_bill", hue='sex', data=tips,
                   estimator=np.median,
                   markers=['o', 'x'], linestyles=['-', '--'], palette="Set2")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303121817.png)  

- 参数说明:
	- estimator: 评估器，默认为计算中值。此处设置计算中位数
	- markers： 如图，点形状
	- linestyles：如图，线型
#### [boxplot](https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot)
boxplot是来表现样本里面的四分位值以及最大最小值的
- Draw a single horizontal boxen plot  

```python
sns.set(style="whitegrid")
ax = sns.boxplot(x="total_bill", data=tips)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303122852.png)

boxplot黑线起点是最小值，终点是最大值。而柱子的起点是25%处的值，终点是75%处的值。柱子中间的那条黑线则对应着50%处的值。跟我们通过df.describe()显示的结果一致

- Draw a vertical boxen plot grouped by a categorical variable

```python
ax = sns.boxplot(x='day', y="total_bill", data=tips, palette="Set3")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303123350.png)


#### [violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot)

violinplot是结合了boxplot和distplot的优点。
通过violinplot既能看到当前样本的最大最小值和四分位值，又能看到对整体分布的预估，了解任意区间的概率分布情况。

- Draw a single horizontal violinplot  

```python
sns.set(style="whitegrid")
ax = sns.violinplot(x="total_bill", data=tips, palette="Set3")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303124549.png)

- Draw split violins to compare the across the hue variable  

```python
ax = sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3", split=True)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303124805.png)

#### [catplot](https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot)
它是以上几种图的接口，以上categorical图表均可通过指定kind参数来[绘制](https://github.com/Vambooo/SeabornCN/blob/master/2%E5%88%86%E7%B1%BB%E5%9B%BE/catplot.ipynb) + FacetGrid

```python
g = sns.catplot(x='day', y='tip', col='sex', data=tips, kind='bar')
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303145952.png)
- 参数说明:
	- col/row: 图表分类依据, names of variables；col横向；row竖向；
	- kind:  “point”, “bar”, “strip”, “swarm”, “box”, “violin”, or “boxen”的一种


### Distribution plots

#### [distplot](https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot)
distplot主要用来对整体分布进行预估，并很容易**观察出某个区间概率的大小情况**。

- Show a default plot with a kernel density estimate and histogram with bin size determined automatically with a reference rule

```python
ax = sns.distplot(a=tips.total_bill)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303125442.png)
distplot展示了整体的分布情况，其中的曲线图则是概率密度函数。
在概率密度函数中，某个点的概率是无意义的。而某两个点之间的概率则是通过对这两个点之间的面积计算得来的。对应到该图上，则意味着total_bill=40的概率是无意义的，但是total_bill在30和40之间的概率是二者之间的曲线下的面积。所以，整个曲线下的面积是1，对应着所有值出现的概率总和。

#### [kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot)
kernel density estimate  

- Plot a basic univariate density
```python
ax = sns.kdeplot(data=tips.total_bill, shade=True, color="r")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303141710.png)

- Plot a bivariate density  
```python
ax = sns.kdeplot(data=tips.total_bill, data2=tips.tip ,)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303142215.png)

- Plot two shaded bivariate densities
```python
iris = sns.load_dataset("iris")
setosa = iris.loc[iris.species == "setosa"]
virginica = iris.loc[iris.species == "virginica"]
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False, cbar=True)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False, cbar=True)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303143237.png)
	- 参数说明:
		- cbar: drawing a bivariate KDE plot, add a colorbar.


### Relational plots

#### [scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html#seaborn.scatterplot)
散点图
- Draw a simple scatter plot between two variables
```python
sns.set(style="darkgrid")
ax = sns.scatterplot(x="total_bill", y="tip", data=tips)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303150606.png)

- Show the grouping variable by varying both color and marker
```python 
sns.set(style="darkgrid")
ax = sns.scatterplot(x="total_bill", y="tip", hue="smoker", data=tips, style="smoker")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303150948.png)
	- 参数说明:
		- hue: 分组变量
		- style: marker变量

- Show a quantitative variable by varying the size of the points
```python
ax = sns.scatterplot(x="total_bill", y="tip", hue="size", data=tips, size="size")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303151448.png)
	- 参数说明:
		- size: 决定圆大小的变量，可以是离散型的

- 聚类结果显示
```python
ax= sns.scatterplot(x='sepal_width', y='sepal_length', data=iris, 
                hue='species', style='species')
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303152012.png)

#### [lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot) 

```python
ax = sns.lineplot(x='tip', y='total_bill', 
                  data=tips, hue="smoker", style="smoker",
                  markers=True, dashes=True, ci=None)
```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303154921.png)

- 参数说明:
	- markers: 实数点 分类标记
	- dashes: 线分类显示（实线、虚线）
	- ci: 置信区间

#### [relplot](https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot)
散点图scatterplot()和折线图lineplot()的接口，散点图和折线图均可通过指定kind参数来绘制 + FacetGrid
```python
sns.set(style="ticks")
g = sns.relplot(x="total_bill", y="tip", col="time", data=tips, kind="line")
g = sns.relplot(x="total_bill", y="tip", col="time", data=tips, hue="smoker", kind="scatter")
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303155523.png)
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303155552.png)

- 参数说明:
	- col/row: 图表分类依据, names of variables；col横向；row竖向；
	- kind: "line", "scatter"

### Regression plots

#### regplot
Plot data and a linear regression model fit.

```python
sns.set(style="darkgrid")
ax = sns.regplot(x='tip', y='total_bill', data=tips)
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303160622.png)

- 参数说明:
	- marker: str, 可设置点的marker; marker="+"


#### [lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html#seaborn.lmplot)
功能比regplot更多一些
Plot data and regression model fits across a FacetGrid
```python
g= sns.lmplot(x='tip', y='total_bill', data=tips, 
              hue='smoker', col='smoker', 
              markers=['o', '+'], palette="Set2")
```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200303161623.png)

### Matrix plots

#### [heatmap(热力图)](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)

```python
pt = iris.corr()   # pt为数据框或者是协方差矩阵
ax = sns.heatmap(pt, annot=True)
```

### Examples
#### highest and lowest
> Identifying counties with highest and lowest Covid-19 Mortality Rates.  

Refer: https://www.kaggle.com/jmarfati/actual-spread-of-covid19-us-county-level-analysis?scriptVersionId=34667620&cellId=23  
![image](https://cdn.jsdelivr.net/gh/xunhs-hosts/pic@master/image.3k61etwq6rs0.png)
```python
# two parallel barplot
plt.figure(figsize=(20,8))

plt.subplot(1, 2, 1)
g=sns.barplot(x='mortality', y='county_state',data=df[df.confirmed>500].sort_values(['mortality'], ascending=False).head(10), color="red")
show_values_on_bars(g, "h", space=0.002, text_size=20)
plt.xlim(0, 0.15)
plt.xlabel("Covid Mortality Rate", size=20)
plt.ylabel(" ", size=20)
plt.yticks(size=15) 

plt.title("Counties with highest Covid Mortality", size=25)

plt.subplot(1, 2, 2)
g=sns.barplot(x='mortality', y='county_state',data=df[df.confirmed>500].sort_values(['mortality'], ascending=True).head(10), color="blue")
show_values_on_bars(g, "h", space=0.002, text_size=20)
plt.xlim(0, 0.05)
plt.xlabel("Covid Mortality Rate", size=20)
plt.ylabel(" ")
plt.yticks(size=15) 
plt.title("Counties with lowest Covid Mortality", size=25)
plt.tight_layout()
```
#### association between two variables
> Association between percentage of population above 65 in the county and covid mortality.

Refer: https://www.kaggle.com/jmarfati/actual-spread-of-covid19-us-county-level-analysis?scriptVersionId=34667620&cellId=27
![image](https://cdn.jsdelivr.net/gh/xunhs-hosts/pic@master/image.6j3aj4motww0.png)
```python
# regplot + barplot(with qcut)
plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
sns.regplot(df_temp.percent_above_65, df_temp.mortality)

plt.subplot(1, 2, 2)
sns.barplot(pd.qcut(df_temp.percent_above_65, 4), df_temp.mortality)
```

---

![](https://cdn.jsdelivr.net/gh/xunhs-hosts/pic@master/jarritos-mexican-soda-KeyMbyMNZNM-unsplash.74pcs7hgodo0.jpg)