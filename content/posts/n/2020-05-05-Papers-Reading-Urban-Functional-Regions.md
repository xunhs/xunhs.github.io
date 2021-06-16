---
layout: post
title: Papers_Reading-Urban Functional Regions
comments: true
toc: true
top: false
categories:
  - 收藏
tags:
  - Points of Interest
  - Urban functional zones
abbrlink: aa427600
date: 2020-05-05T06:40:39+00:00
---

> 


<!--more-->



#### Journal Article: 2017-Hierarchical semantic cognition for urban functional zones with VHR satellite images and POI data 
- Refer: [ISPRS Journal of Photogrammetry and Remote Sensing](http://dx.doi.org/10.1016/j.isprsjprs.2017.09.007)
- A
- Abstract
  - [功能区划图不好拿]functional-zone maps are hardly available in most cities
  - [急需(半)自动化的方法]an automatic/semi-automatic method for mapping urban functional zones is highly required
  - [继承性语义识别]Hierarchical semantic cognition (HSC) 
  - relies on geographic cognition and considers four semantic layers
  - with a very-highresolution (VHR) satellite image and point-of-interest (POI) data
  - result:  overall accuracy of 90.8%; the contributions of diverse semantic layers are quantified


#### Journal Article: 2018-Understanding Urban Functionality from POI Space
- Refer: [2018 26th International Conference on Geoinformatics](https://ieeexplore.ieee.org/abstract/document/8557122/)
- A
- Abstract:
  - understanding of the urban built environment
  - revealing the co-occurrences of POIs 
  - POI Space
  - the network of relatedness between POIs
  - findings:
    - [核心-边缘分布]more common POIs are located in a densely connected core whereas rarer and more unique POIs occupy a less-connected periphery
    - [扩散速度?]common POIs act more on the speed of diffusion, unique POIs act more on the scope of diffusion.


#### Journal Article: 2019-Beyond Word2vec: An approach for urban functional region extraction and identification by combining Place2vec and POIs



#### Journal Article: 2019-DFCNN-Based Semantic Recognition of Urban Functional Zones by Integrating Remote Sensing Data and POI Data
- Refer: [Remote Sensing](https://www.mdpi.com/2072-4292/12/7/1088)
- A
- recognition of physical and social semantics of **buildings**
- object-wise recognition strategy
- building semantic recognition



#### Journal Article: 2020-Understanding Place Characteristics in Geographic Contexts through Graph Convolutional Neural Networks
- Refer: [Annals of the American Association of Geographers](https://www.tandfonline.com/doi/full/10.1080/24694452.2019.1694403)
- **AAAAA**
- Place Characteristics; Geographic Contexts; Graph convolutional neural networks (GCNNs) 
- Abstract:
  - both its observed attributes and the characteristics of the places to which it is connected
  -  spatial prediction task: predict the unobserved place characteristics based on **the observed properties and specific place connections**
  -  GCNNs capture the knowledge of the relevant geographic context
  -  A series of comparative experiments
  -  formalizing places for **geographic knowledge representation and reasoning**

- Introduction
  - place characteristics
  - places are not isolated but are connecthsed ti each other
  - the contextual information for a place (i.e., its connection to other places) is crucial to understand its characteristics
  - place **conncetions** => the measures between places (distance, adjacency and spatial interactions)
  - [为什么会提到地理空间层次的上下文呢？我理解，正如作者所言，对位置地点属性的预测，不仅仅依赖于该地点的观测变量，同时还由该地点周边/相连接的地点的观测属性决定。这里的周边/相连接，对应着作者论述的地理空间上下文]geographic contexts => The prediction of a place’s unknown characteristic relies on both the place’s observed characteristics and **the characteristics of the places to which it is connected**. 
  - [这里引入了GCNN,提到几个关键词:aggregation,neighbors,contextual infformation]process **the connection information**: GCNNs generally follow an **aggregation scheme** where each node aggregates characteristics of its neighbors to learn a deep representation of the contextual information
  - each place is represented as a node, place characteristics are the node features to be computed, and place connections are represented as the graph edges
  - Introduction部分可以说`短小精悍`了,内容不多但是论点阐述的很清楚。
    - 第一段通过place引入place characteristic的概念，为后面做铺垫；
    - 第二段说place不是孤立的而是相连的，引入了place connection的概念；
    - 第三段就用到了上面两个概念的铺垫了，他说place characteristic的预测不仅和自身的观测变量有关，还和相邻的(connected)的地点的特征相关，然后介绍了两个measure connection的研究。然后就是说道研究的局限性，局限性其实他表述了比较多的方面，也可能是我理解的比较抽象，`这一段的内容可能比较关键,因为他把两个概念穿了起来，并且引出了本文的研究点`；
    - 第四段理解上就比较简单写了，引入GCNN对于解决model connection的问题很有效；
    - 第五段简介自己的研究内容。

- Methodology
  - Building the Place-Based Graph
  - Predicting Place Characteristics Using GCNNs
- Case Study
  - Study Area
  - Data Preparation
    - Delineating Place Boundaries.
    - Quantifying Place Characteristics.
    - Quantifying Place Characteristics.
  - A GCNN Model to Predict Places’ Functional Features
- 我把他方法论和Case Study的部分列出来是想说他这两部分的划分我有点看不懂。方法论部分提取出来，然后Case Study去讲具体的步骤。通常的文章里面都不分开吧？或者具体的步骤放在implementation里面？


#### Journal Article: 2020-Urban Function as a New Perspective for Adaptive Street Quality Assessment
- Refer: [Sustainability](http://dx.doi.org/10.3390/su12041296)
- A
- Abstract
  - Street Quality Assessment => managing natural and public resources, organizing urban morphologies and improving city vitality
  - from the perspective of the variation in urban functions
  - urban function detection + urban function-driven multilevel street quality assessment
- Introduction 
  - assess street networks => enriches the current description of **street networks** and enhances the evaluation of street network performance
  - these studies have discussed greenery, mobility patterns, and land-use connectivity but **ignored the different urban functions** that each type of street serves
  - [静态的？]the detection of urban functions in most research is static
  - commercial, residential and traffic functions

***

![](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/usr/uploads/2020/20200505120208.png)


