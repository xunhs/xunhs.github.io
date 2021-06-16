---
title: Papers Reading:Urban functions/Urban land use
author: Ethan
type: post
date: 2020-10-21T09:30:38+00:00
url: /2020/10/21/102/
argon_hide_readingtime:
  - 'false'
argon_meta_simple:
  - 'false'
argon_first_image_as_thumbnail:
  - default
views:
  - 30
categories:
  - 收藏
tags:
  - 城市功能区
  - 特征融合
  - 论文阅读

---
> 文献阅读：城市功能区划分与城市土地利用分类相关。主题为遥感数据与社交型大数据融合方法。


<!--more-->

### 2016 - Mapping Urban Land Use by Using Landsat Images and Open Social Data

- remote sensing
- https://doi.org/10.3390/rs8020151

#### 札记

- Introduction 讲的城市化进程和land use、land cover的差异写的不错，值得借鉴参考
- Introduction 他一直在讲 `relatively large spatial scale`
- Introduction 还是从遥感的角度来讲的，open social data的戏份不多；而且有种虎头蛇尾的感觉
- Two-steps classification: 先区分建成区和非建成区，在此基础上两类分类

#### 标签
- RS + POI

#### Abstract

- **high-resolution** urban land use maps have important applications
- the **availability** of these maps is low in countries such as China
- using **satellite images and open social data**
  - used **10 features** derived from Points of Interest (POI) data 
  - and **two indices** obtained from Landsat 8 Operational Land Imager(OLI) images
-  classify parcels into **eight Level I classes and sixteen Level II classes** of land use (两级分类)
- tested in **Beijing**, China

#### Introduction

- 城市化的背景（可以借鉴参考）：<u>Urbanization</u> in China => <u>Large-scale</u> urbanization has had a dramatic <u>impact</u> => <u>Studies that assess this process and its impacts</u> are important => To achieve these goals, <u>detailed urban land cover/use maps are required</u>
- [land use和land cover的差异] Currently, land cover information is a **primary data source** => detailed information on **urban land use** is needed
  - land use is a <u>cultural concept</u> that describes human activities and their use of land
  - whereas land cover is a <u>physical description</u> of land surface
  - <u>land cover can be used to **infer** land use</u>, but the two concepts are not entirely interchangeable.
- 然而，
  - 问题1：high-resolution urban land use maps covering large spatial extents are relatively **rare**
  - 原因-发展中国家认知和技术不行：<u>local knowledge and the techniques necessary for developing these types of maps are often not available</u>, particularly for **developing regions**
  - 问题2：**outdated** maps
  - 原因：urban land use maps are normally <u>produced by interpreting aerial photographs, field survey results, and auxiliary materials</u>, such as appraisal records or statistical data
  - 因此：As a result, to obtain land use maps that capture the pace of urban development in a timely and accurate manner at a <u>relatively large spatial scale</u> is a **critical challenge** in urban studies, both in China and in other countries facing similar situations.
- **Satellite-based remote sensing** holds certain advantages
  - large spatial coverage, high time resolution, and wide availability
  -  pixel-based image classification methods: using spectral and/or textural properties
  -  per-field and object-based classification methods
  -  **medium-resolution** satellite images
    - difficult to extract socioeconomic features of urban areas
    - cannot provide sufficient separation among urban functional zones
  -  **high spatial and spectral resolution** satellite images
    - provide more detailed information on urban structures
    - 贵: prohibitively expensive in general
-  emergence of **open social data** => mapping urban land uses at <u>high-resolution</u>
  - containing spatiotemporal patterns of human activities
  - the existing studies were <u>often implemented over relatively small areas or specific land use types</u> using data sets that were <u>subjective or proprietary</u>(主观)
  - the physical attributes of urban functional parcels were <u>seldom</u> included
- **integrating** social knowledge with remotely sensed data
  - physical features extracted from satellite data
  - socioeconomic features retrieved from open social data
  - One type of open social data that is particularly **promising for this purpose** are <u>Points of Interest (POI) data</u>.
  - [就当时而言吧] As far as we know, <u>there are no reports</u> on using POI data and satellite data <u>jointly</u> to produce detailed land use maps.
  - 我们的工作: developed a protocol that utilizes medium-resolution satellite images and POI data to map urban land uses.

#### 数据和方法

- Method![image-20201022101428777](https://i.loli.net/2020/10/22/bUwrhaARnf5Xy6F.png)
  - 两步划分
  - [先划分为建成区和非建成区] differentiate the <u>built-up and non-built-up areas</u> based on classified impervious surface(不透水面) areas
  -  the classified built-up and non-built-up regions were <u>merged into a final land use map</u>
- Classification System - 分类系统
  -  **built-up area** as places dominated by <u>artificial buildings and structures</u>
  - **non-built-up area** as places mainly occupied by <u>cultivated land, forests, grassland, water and water conservancy facilities</u>
  - two-level classes![image-20201022101845180](https://i.loli.net/2020/10/22/UKAI8EDu9FGTca7.png)

### 2017 - Classifying urban land use by integrating remote sensing and social media data

- International Journal of Geographical Information Science
- https://doi.org/10.1080/13658816.2017.1324976

#### 札记

- groud truth是通过人工目视解译打的标签
- 工作流还需要很多手工调参的地方，比如kmeans等
- 使用的基础特征，如low-level的visual features和frequent features 没有充分利用数据带来的特征。

#### 标签

- RS与多源社交媒体融合
- 融合方法：土地利用词典

#### Abstract

- accurate classification of urban functional zones
- urban land use classification by considering features that are extracted from **either** <u>high spatial resolution (HSR) remote sensing images</u> **or** <u>social media data</u>, but few studies consider **both** features due to the lack of available models.
- Proposed: a novel **scene classification** framework to identify **dominant** urban land use type **at the level of traffic analysis zone** by <u>integrating probabilistic topic models and support vector machine</u>
- A land use word dictionary
- fusing:
  - **natural–physical features** from HSR images
  - **socioeconomic semantic features** from multisource social media data
- comparing with **manual** interpretation (人工目视解译) data

#### Introduction

- 背景论述: Land use and land cover (LULC) information => urban functional zones which reflect in urban land use patterns => the effective detection of **urban land use patterns**
- High spatial resolution (**HSR**) remote sensing images
  - computation-based urban land use detection
  - three types of spatial units
    - <u>units of pixels and objects</u> are usually employed to evaluate land cover,
    - whereas <u>scenes</u> are commonly used to identify urban functional zones and accurate urban land use patterns
  - object-oriented classification(**OOC**) models
    - physical features (such as spectral, shape, and texture features) of ground components
    - 缺陷: often <u>overlook the spatial distribution and semantic features of ground components</u> because they were
      <u>only designed to mine the low-level semantic land cover information</u> of ground components
- Semantic Gap
  - **low-level semantic features** indicate ‘information’ that comes with the data directly
  - **high-level semantic features** refer to ‘knowledge’ specific for each user and application
  - semantic gap refers to the **disparity** (不一致) of features identified between these two levels
- To **bridge** the ‘Semantic Gap’ between LULC, recent studies have introduced the concept of ‘**scene classification**’ into HSR image classification to <u>label a scene with a single category</u>
  - Current studies: apply the <u>bag-ofwords (BoW)</u> modeling approach and fuse physical features of ground scenes via
    <u>probabilistic topic models (PTMs)</u>
  - However, extracting features from remote sensing images can <u>only represent the **external** (外部的) natural–physical properties of ground components</u>, whereas **regional** land use types often have a strong correlation with **indoor human socioeconomic activities**, which are difficult to extract from HSR images.
- To solve this problem, recent studies have proposed the concepts of <u>‘social sensing’ and ‘urban computing’</u>
  - Multisource social media data => <u>monitor</u> residential activities and urban land use dynamics
- However, these methods <u>utilize **only one type of data** rather than f**using geospatial information**</u> from HSR images and social media data into the detection of land use types.
  - 假设-同类区域同特征: Regions with similar types of urban land use tend to have similar external natural–physical properties and indoor human socioeconomic activity patterns
- 本文工作:
  - aims to build a **dominant** urban land use sensing framework by
  - combining several machine learning and natural language processing (NLP) **models** to fuse the geospatial latent semantic information extracted from HSR images (**remote sensing information**) and multisource social media data (**social sensing information**) as patterns
  - to classify the urban land use and evaluate the accuracies and reliabilities of classification models by **manual** interpretation
  - **Haizhu district** in Guangzhou

#### 数据及方法

- 遥感数据
  - high spatial resolution (HSR) Worldview-2 image
  - 2014，grid size of 34,263 × 14,382 and a spatial resolution of 0.5 m
  - Taz换分为 593 land patches
  - 手工目视解译打的标签，包含**七类**: public management services land (M), industrial land (I), green land (G), commercial land(C), residential land (R), park land (P),  and urban villages (U).
- 社交媒体数据
  - OpenStreetMap (OSM) road networks
  - Gaode POIs (说是用API获取的，没说时间)
  - real-time Tencent user density (RTUD)：spatial resolution of 25 m
- Method

  ![image-20201021205454501](https://i.loli.net/2020/10/21/SNCGx4yvmIszbnf.png)

  - K-Means需要手工定义参数；使用low-level的visual features；
  - 一股脑都当做单词了



### 2017-Hierarchical semantic cognition for urban functional zones with VHR satellite images and POI data 
- ISPRS Journal of Photogrammetry and Remote Sensing
- http://dx.doi.org/10.1016/j.isprsjprs.2017.09.007
- 张修远

#### 札记

#### 标签

#### Abstract

- [功能区划图不好拿]functional-zone maps are hardly available in most cities
- [急需(半)自动化的方法]an automatic/semi-automatic method for mapping urban functional zones is highly required
- [继承性语义识别]Hierarchical semantic cognition (HSC)
- relies on geographic cognition and considers four semantic layers
- with a very-highresolution (VHR) satellite image and point-of-interest (POI) data
- result:  overall accuracy of 90.8%; the contributions of diverse semantic layers are quantified

### 2018 - Integrating Aerial and Street View Images for Urban Land Use Classification

- remote sensing
- https://doi.org/10.3390/rs10101553

#### 札记

- 街景空间对齐的方式
- feature maps stacking
- 实验明明做出来是遥感影像语义分割的结果呀~感觉是套上Urban land use的壳，做着语义分割的活
- 特征自对比；没有与其他方法/baseline的对比

#### 标签

- Aerial and Street View Images
- Pixel-level land use classification
- SegNet based Encoder-Decoder
- 端到端

#### Abstract & Introduction

- Urban land use
- rely heavily on domain experts
- in this paper
  - deep neural network-based approaches
  - label urban land use at pixel level
  - using high-resolution aerial images and ground-level street view images
- specifically
  - **extract semantic features** from sparsely distributed street view images
  - **interpolate**(插值) them in the spatial domain to match the spatial resolution of the aerial images
  - **fused** together through a deep neural network for classifying land use categories
  - tested on a large publicly available aerial and street view images dataset of **New York City**
- Results
  - using aerial images **alone** can achieve relatively high classification accuracy
  - the ground-level street view images **contain** useful information for urban land use classification
  - fusing street image features with aerial images can **improve** classification accuracy
  - street view images **add more values** when the resolutions of the aerial images are lower
- Contributions
  - presents a novel method to **fuse** extracted ground-level features from street view images with high-resolution aerial images to <u>enhance pixel-level urban land use classification accuracy</u>; **two sources of images** collected from totally **different views** (i.e., overhead and ground-level views)
  - examines the <u>impact of aerial image resolution changes</u> on classification accuracy; investigate into the <u>contribution</u>
    <u>that street view images make to the improvement</u> of the classification results
  - explore <u>deep neural network methods</u> for <u>pixel-level urban land use classification</u>  => enriches the remote sensing applications

#### Method

- Ground Feature Map Construction （怎样特征空间对齐的）![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201030162224.png)
  - semantic feature extraction
    - pretrained **Places-CNN** (without the last fully connection layer)  => extract a **512**-dimensional feature vector for each **image**
    - **concatenate** the extracted four feature vectors into a **2048**-dimensional feature vector for each **location**
    - principal component analysis (PCA)  is used to <u>compress semantic information and reduce the dimension</u> of the feature vector to **50**
  - spatial interpolation
    - 起因：places with street view images are **sparsely distributed along roads** in the spatial domain
    - street view images <u>capture the scenes of **nearby** visual areas</u> instead of single dots in the space [街景捕捉周边的场景]
    - 因此，**project** the semantic information of street view images to their **covered areas** from top-down viewpoint
    - 策略：
      - use spatial interpolation method - Nadaraya-Watson kernel regression
      - estimate the impact of nearby street view images on a pixel - Gaussian kernel to calculate weights
- Data Fusion (怎样融合训练的)![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201030163430.png)
  - two encoders: feature maps stacking
  - the output feature map is fed to a Softmax layer to make **pixel-level** predictions



### 2019-Beyond Word2vec: An approach for urban functional region extraction and identification by combining Place2vec and POIs

### 2019 - Model Fusion for Building Type Classification from Aerial and Street View Images

- remote sensing [半个月就接收了！不可思议，我的小修文章半个月还在编辑手里呢 Orz]
- https://doi.org/10.3390/rs11111259

#### 札记

- 主要是做的用Google影像+街景识别建筑物功能，重点讲的在模型融合方面
- 研究现状讲的是urban land use with DL 做了一个summerized的表格 还行 写现状的时候可以回过来参考
- 结论是结论层的融合（虽然简单）比特征层的融合效果好，并简单分析了下原因

#### 标签

- Aerial and Street View Images
- Building Type Classification
- 融合策略：decision-level fusion vs feature-level fusion
- 端到端

#### Abstract

-  **mapping building functions** <u>jointly</u> using both **aerial and street view images** via **deep learning** techniques
- **a data fusion strategy** => cope with **heterogeneous** image modalities (形态)
- <u>geometric combinations of the features</u> (图像的特征的几何组合) of such two types of images, especially <u>in an early stage of the convolutional layers</u>, often lead to a **destructive effect** (破坏性的效果) due to <u>the spatial misalignment of the features</u> (特征的空间错位).
- proposed a **decision-level fusion** of a diverse ensemble of models (compared to the <u>common multi-stream end-to-end fusion</u> approaches)
-  sophisticated classification schemes => highly **compact classification scheme** with four classes, commercial, residential, public, and industrial

#### Introduction

- compared two model fusion strategies (two-stream end-to-end fusion network)
  -  a **geometric**-level model fusion
  - **decision**-level model fusion
- A summary of the models and fusion strategies

#### Data and Method

- Data
  - we extracted **geolocation** and the attributions of **building** function annotated by volunteers from **OSM**.
  - Then, the <u>associated street view images and the overhead remote sensing images</u> of each building instance were retrieved via BingMap API and Google Street View API using its geolocation![](https://i.loli.net/2020/10/22/6o5WLpG7riBmRJx.png)

- Building Classification Scheme/System
  - extracted t<u>he class of each building</u> from the volunteered building tag from OSM
  - <u>selected the 16 most frequently</u> occurring building tags in our raw dataset and <u>aggregated them into four cluster classes</u>
  - follow <u>a very basic but widely accepted</u> classification scheme with four classes: **commercial, industrial, public, and residential**![image-20201022215334882](https://i.loli.net/2020/10/22/PbrM5nyXWKYQaNI.png)

  - Data Volume: **56,259 buildings** with four images for each building (区域党？？为什么要分区域划分数据集呢？)
    - the images from the state of Wisconsin and Wyoming were used as validation samples (1943 buildings)
    - Washington and West Virginia were used as test samples (2212 buildings)
    - remaining 47 areas were used as training samples (52,104 buildings)

- Method
  - existing deep neural networks pre-trained on very large datasets: **Places365** and ImageNet
  - **two-stream** **end-to-end** fusing networks![image-20201022215709070](https://i.loli.net/2020/10/22/qWG9xK7UTCsoVc3.png)
  - decision-level fusion
    - combines the <u>softmax probabilities or the classification labels</u> directly
    - model blending and model stacking![image-20201022215903182](https://i.loli.net/2020/10/22/oBWVbpFgXr3EkeY.png)
  - 他的Experiments and Discussion（包含训练参数和训练过程）可以参考下，以后训练的经验
#### Conclusion
- **a decision-level fusion** of street view and overhead images often **outperforms** a feature-level fusion, despite its simplicity
- we employed decision-level fusion strategies to achieve great performance <u>without significantly altering the current network architecture</u>
### 2020 - Deep learning-based remote and social sensing data fusion for urban region function recognition
- ISPRS Journal of Photogrammetry and Remote Sensing
- https://doi.org/10.1016/j.isprsjprs.2020.02.014

#### 札记

- Introduction写的确实好，感觉没有一句废话。开头的背景介绍和方法、最后的文章总结都可以用作模板。
- 很好的方法类论文模板

#### 标签
- 多模态：RS + 时间序列数据
- 端到端
- 改进损失函数

#### Abstract
- Urban region function recognition
- effectively **integrating** the multi-source and multi-modal remote(多模态) and social sensing data remains technically challenging
-  **end-to-end** deep learningbased remote and social sensing data fusion model
- two data sources are **asynchronous** (异步的)
-  simultaneously optimizing three costs
  - classification cost
  - cross-modal feature consistency (CMFC) cost
  - cross-modal triplet (CMT) cost
- conducted on publicly available datasets: [百度AI-城市区域功能分类](https://aistudio.baidu.com/aistudio/competition/detail/13); [飞桨官方基线](https://aistudio.baidu.com/aistudio/projectDetail/176495)
- The results show that the seemingly **unrelated** physically sensed image data and social activities sensed signatures can indeed **complement** each other to help enhance the accuracy of urban region function recognition

#### Introduction
- 背景 - [从城市区域有限，城市人口膨胀角度论述背景，可参考] it is of great importance to monitor and manage the **limited** urban areas for such a huge population
- 研究问题 - urban region function recognition **VS** land use and land cover (LULC) classification (各有各的说法吧)
  - LULC stresses on <u>physical characteristics of the earth surface</u>
  - urban region function recognition focuses purely on <u>socioeconomic functional attributes of urban regions</u>
  - region function recognition using remote sensing images <u>**alone** is not sufficient</u>
- 数据 - [这一段写的蛮好，逻辑性也很强]
  - **social sensing big data**
    - by-products of human daily life;
    - contain rich socioeconomic attributes
  - <u>When these data meet with remote sensing, **the promising trend** is to fuse them to recognize urban functions, since the two kinds of data are  complementary to each other</u> (这句感觉写的蛮好)
  - [问题来了] remote and social sensing data are significantly **different in terms of sources and modalities** 
  - [关键在于解决这个问题] The key challenge is to **alleviate the modality gap and heterogeneity** between them 
- 方法 - deep learning
  - powerful abilities to automatically learn **high-level features** from large amount of data
- 本文工作及贡献
  - [设计模型] end-to-end deep multi-modal fusion method
  - [两种处理网络] two effective neural networks to extract temporal signature features automatically
  - [两个costs] two auxiliary losses => address the data asynchronous problem
  - [设计实验并分析] conducted extensive experiments on open available datasets;  analyze and discuss the results thoroughly to give insights into fusing the two multi-modal data
- 相关工作部分一些总结
  - Remote and social sensing data are of significantly different sources and modalities, they <u>possess different information about urban land surface and are complementary</u> to each other
  - Most existing works use **handcrafted** features (手工特征), which require <u>human experts and are laboriou</u>s.

#### 方法和数据
- Framework![](https://i.loli.net/2020/10/25/jFBIbqpHtLan39C.png)
  - image encoder  => extract time-dependent features from time-series signature data
  - temporal signature (TS) encoder
  - Fusion methods: concatenation, element-wise sum, and element-wise max pooling.
  - the fused feature is then fed into fully connected layers and softmax layer to make the final prediction.
  - Loss functions
    - the major loss: cross entropy loss
    - the auxiliary loss:
      - [两类特征尽量相似(cos相似性)] cross-modal feature consistency (CMFC) loss: Analogous to document alignment, since both the image and signature data are indicative of (象征) the same urban function properties of the same region, <u>there ought to be **correlation** between them in spite of different modalities</u>.  The CMFC loss <u>enforces the features of image and signature to be **consistent and similar** with regard to vector orientation</u>.
      - [[三元组triplet损失函数](https://blog.csdn.net/zenglaoshi/article/details/106928204)] cross-modal triplet (CMT) loss: further utilizes the <u>category information</u> and tries to draw cross-modal features of the same class **nearer**, while push features of different classes **far away**

### 2020 - Recognizing urban functional zones by a hierarchical fusion method considering landscape features and human activities
- Transactions in GIS
- https://doi.org/10.1111/tgis.12642

#### 札记
- Introduction套路有点熟，而且写得感觉有一点乱
- 所以方法的最后一步Recognizing到底是怎么做的呢？就是普通的叠加产生复杂区么？
- 创新的地方个人感觉满牵强 能写出来也不容易

#### 标签
- RS + POI + Taxi GPS Trajectories
- hierarchical fusion method

#### Abstract
- functional zones
- two basic factors
  - urban landscape environment: provides the **basic space** for human activity and influences urban land use at a **coarse** scale
  - human activities: indicates the **differentiation of functions** in local urban areas
- In previous studies, <u>**the hierarchical correspondence and interaction** (层次对应关系和相互作用) between urban landscape and human activities **have not been given full consideration in**</u> the cognition of urban functional zones, which would influence the accuracy and interpretability of the results.
- a hierarchical **fusion** method
  - a land use classification based on urban landscape features from <u>remote sensing images</u>
  - fine‐grained **functional semantics** of local urban areas are recognized based on human activity patterns extracted from crowdsourced data (i.e., <u>points of interest and taxi trajectories</u>)
  -  the above results at different scales are fused with **hierarchical constraints**
- Results
  - Wuhan, China
  - overall accuracy of the proposed method is 82.51% (accuracies of the mixed functional zones and single functional zones are 77.93 and 87.96%, respectively)
  - compared with state‐of‐the‐art methods, the proposed method performed better for the recognition of **mixed functional zones**

#### Introduction
- 第一段
  - urban functional zone定义
  - concept of functional zones与两个因素密切相关：
    - urban landscape: determines the basic type of urban land use at a coarse
      scale (such as water, forest, built-up areas)
    - human activities: indicate the differentiation of functions in finegrained urban regions (e.g. the built-up area at a coarse scale may consist of residential, commercial, industrial, or mixed land use zones) 
  - urban functional zone maps的重要性 => the functional zone maps are still **hardly available** in many cities
  - The recognition of urban functional zones的意义
  - <u>Numerous methods</u> have been developed to cognize the functional zones based on <u>remote sensing images and crowdsourced data</u>
    - Remote sensing images
      - one of the commonly used data sources to detect urban land cover objects and land use classification
      - forming the basis of semantic cognition of urban functional zones
- 第二段
  - many studies based on remote sensing images
  - most of these methods are based on the <u>physical properties</u> (such as spectral and textural features) of the objects, which makes it <u>difficult to disclose the functional attributes of the urban areas</u>
    - 例子: **buildings** in different urban areas may have **different functions**, such as residential, educational, and commercial buildings, which are **hard to differentiate based only on the physical properties** from remote sensing images 
  - 引出多源数据：it is essential to work with **multiple data sources** for the recognition of urban functional zones, especially the crowdsourced data related to human activities
- 第三段
  - crowdsourced data
  - Since <u>different kinds of data can reflect different characteristics</u> of urban functional zones, it is essential to <u>integrate multi-source data</u> to improve the accuracy of recognition of functional zones 
- 第四段
  - 仍有提升的空间 Although <u>many methods have been proposed</u> for discovering urban functional zones, there is still <u>space for improvement</u>.
  - From the perspective of urban morphology=> the formation of urban functional zones => both the underlying landscape environment and human activities in urban space
  - Most of the existing studies only <u>focus on one aspect and neglect the integration</u> of landscape and humanity
  - in this paper => a hierarchical fusion approach
    - a new area-weighted proportion of POIs
    - introduced information entropy  => the combinational diversity of mixed functions
    - comprehensive: integrating the underlying <u>landscape features, spatial pattern of ground objects, and human activity patterns</u>

#### 方法和数据
- 数据
  - land patches: divided the area into **915 zones** using the road network and the <u>morphological partition algorithm</u> of Yuan et al. (2012)
  - Landsat-8 images downloaded from the Geospatial Data Cloud (http://www.gscloud.cn/sources) 
  - POIs: Gaode map
  - taxi GPS trajectories: 8,141 taxies on May 8–14, 2017, with a total of 1.65 million trajectories
- Methodology
  - 三部分：urban landscape classification, functional semantic feature extraction, and functional zone recognition  ![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227190909.png)
  - urban landscape classification
    - Remote sensing images
      - reflect the real state of the urban land cover, forming the basis of urban land use
      - classify the urban landscape into different land use categories  => “woodland,” “farmland,” “water,” and “built-up” areas
      - spectral, texture, and spatial features
        - **spectral descriptor**: Mean, standard deviation, minimum, and maximum values of spectral features in a moving window (25 × 25 pixels) for each band of the image
        - texture features: The gray-level co-occurrence matrix (GLCM)
        - texture descriptor: four Haralick texture measures, including contrast, sum average, variance, and entropy in a moving window
        - spatial descriptor: The area, compactness, and convexity measures
  - human activity features extraction
    - land use functional semantics extraction using a vote-based model
      - aggregated these POI types into six basic functional categories according to the Standard for Urban Land Classification and Land Use Planning for Construction **(GB50137-2011)**   【这个图蛮好看的，以后可以参考】![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227160101.png)
      - vote-based model to POIs
        - area-weighted proportion (AWP) of POIs  【POI不是点么，有面积么】
        - Shannon's information entropy => measure the mixed degree of functional categories
        - 阈值区分单一类型和复杂功能区
    - Human mobility patterns detection and function inference
      - POI => only reveal the **static semantic information** of human activities
      - Taxi GPS trajectory => dynamic human mobility patterns
      - origin–destination (OD) flows => **24-dimensional feature vector (F)**
        - divided a day into 12 time periods
        - F={O, D}
        - normalized volumes
      - the fast nearest-neighbor (NN) classification model
        - the Pearson correlation coefficient is used to measure the similarity of feature vectors (F) of two urban zones 
        - manually labeled the functional categories of 915 urban zones
  - Functional zone recognition by hierarchical semantic fusion【图参考】![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227194103.png)
  - Evaluation of results
    - the overall accuracy (OA)
    - kappa coefficient
#### Results

![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227195431.png)
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227195709.png)
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227195734.png)
![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201227195802.png)

### 2020-Mapping essential urban land use categories in China (EULUC-China):preliminary results for 2018
- 宫鹏
- Science Bulletin
- Short Communication
- https://doi.org/10.1016/j.scib.2019.12.007
#### 札记
- preliminary/initial results：作者强调初步的分类结果，意思你懂吧（我的结果、精度可能不好，但是我是第一个做大尺度的）
- 分类体系：Essential(基本) Urban Land Use Categories, EULUC （可以参考）
- 分类结果共享了，在这里：http://data.ess.tsinghua.edu.cn/
#### 标签
- urban land use map for entire China
#### 内容
- 第一段
  - Land use => Urban land use => widespread effects
  - 目前难题：
  	- **maps, pattern and composition** of various land use types in urban areas, are **limited** to city level
  	- The mapping standard on data sources, methods, land use classification schemes **varies** from city to city
	- 挑战：<u>various national and global environmental challenges caused by urbanization</u>
  - =>急需方案：urban land uses at the national and **global scales** that are derived from the **same or consistent data sources** with the **same or compatible classification systems** and mapping methods
  - 然而，现状是：
  	- a number of **urban-extent maps exist** at global scales, more **detailed urban land use maps do not exist** at the same scale
	- consistent **land use mapping efforts** are rare
- 第二段
	- 城市土地利扩展 => 城市不透水面
	- 再次强调难题：However, we do not have a **complete knowledge about the distribution, pattern and composition of more detailed land use types** in urban China
	- 现有工作：more and more **efforts** have been made to <u>map individual cities</u> using a combination of remotely sensed data and open social data
- 第三段，本文工作：=> a new urban land use map for entire China
	- **input features**: 10-m satellite images, OpenStreetMap, nighttime lights, POI and Tencent social big data all in 2018
	- **A two-level classification system**: Essential Urban Land Use Categories, EULUC![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201229221526.png)
	- **Training and validation samples** are separately collected through a crowdsourcing approach
	- present the **initial results** for producing EULUC of China
- 第四段，步骤简述，Four major procedures：![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20201229221613.png)
	- parcel generation with OSM road network and impervious boundaries
	- feature extraction from multisource geospatial big data
	- crowdsourcing collection of training and validation samples
	- mapping and accuracy assessment of EULUC-China
- 第五段，polygon-based parcels， 交通分析小区的生成细节
	- represents a **relatively homogeneous socioeconomic function**(相对同质)
	- polygons bounded by road networks
	- backbone: buffered major roads and minor roads
	- overlaying the parcel map with the impervious boundaries and the water layer
	- => 440,798 urban parcels
- 第六段，Four types of features，输入特征的细节（数据：特征）
	- 10-m Sentinel-2A/B images: mean and standard deviations of blue, green, red, and near-infrared bands, normalized difference vegetation index, and normalized water index
	- Tencent mobilephone locating-request (MPL) data: 8-h mean trajectories of the active population during weekdays and weekends
	- 130-m Luojia-1 nighttime light images: mean value of digital number
	- Gaode POI data: total number of all POIs, and total number and proportion of each type of POIs within each parcel
- 第七段，crowdsourcing campaign for collecting training and validation samples，做标签，训练集、验证集
	- 21 research groups in 27 selected representative cities
	- training sample collection
		- The selected training parcels are required to be **typical and stable with low mixing of land uses**
	- validation sample collection
		- **randomly** generate around 50 parcels in each city
		- on-site survey: geolocations, Level I and II categories, landmark buildings and facilities, and mixed land use situation and their estimated proportions
	- 1795 training sample parcels and 869 validation sample parcels in total with a high confidence
- 第八段，模型训练
 - produce a parcel-level mapping of EULUC-China with the random forest (RF) classifier
 - transportation lands样本量太少，评价的时候排除
- 第九段，结果简述
	- R, C, P => clustered in urban cores
	- I => distributed in suburban areas
	- each city has its respective characteristics
	- Statistically (参考分析结果)
		- within the 166,338 km2 impervious area of China in 2018
		- residual lands account for 25.0%(41,576 km2)
		- commercial lands account for 4.4% (7317 km2)
		- industrial lands account for 40.6% (67,588 km2)
		- transportation lands account for 11.2% (18,576 km2)
		- public management and service lands account for 18.8% (31,281 km2)
- 第十段，全局精度评价
	- independent validation sample is 61.2% for Level I and 57.5% for Level II
	- **overall accuracy varies** from 40.4% to 82.9% for Level I, and from 34.0% to 80.0% for Level II
	- complexity of parcel-level land use, land use mixture => an impact on the performance of the validation process
	- RETIO: Residential, Entertainment, Transportation, Industrial, and Office
- 第十一段， the contribution of different types of features, 特征贡献率
	- Compared with the use of POIs only: remotely sensed data and social big data help further improve the overall classification accuracy (~7%)
	- quantify the variable importance in random forest model in terms of mean decrease of accuracy and mean decrease of Gini coefficient => **POI has the greatest contribution, followed by Sentinel-2 multispectral features, Luojia-1 and Tencent features**
- 第十二段，综合评价及展望
	- **mark the beginning** of a new way of **collaborative** urban land use mapping **over large areas**
	- weakness of crowdsourcing
		- **斑块划分方式改进，quality improvement of parcels**: **finer division of existing parcels** with the help of more detailed road networks and image segmentation techniques has good potential => 更加精细的斑块划分
		- **更为系统的采样、特征及算法**，systematic testing of samples, features, and algorithms: Knowledge about the impact of sample size and feature combinations on classification performance
		- **软分类策略，hard and soft classification strategies**: expand classification from the current **dominant-class only** to **multiple-class per-parcel classification**

###  2020-DFCNN-Based Semantic Recognition of Urban Functional Zones by Integrating Remote Sensing Data and POI Data
- Remote Sensing
- https://www.mdpi.com/2072-4292/12/7/1088

###  2020-Understanding Place Characteristics in Geographic Contexts through Graph Convolutional Neural Networks
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
  - A GCNN Model to Predict Places’ Functional Features
- 我把他方法论和Case Study的部分列出来是想说他这两部分的划分我有点看不懂。方法论部分提取出来，然后Case Study去讲具体的步骤。通常的文章里面都不分开吧？或者具体的步骤放在implementation里面？


###  2020-Urban Function as a New Perspective for Adaptive Street Quality Assessment
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
<!-- 插入图片 -->
![](https://i.loli.net/2020/10/21/EjPFLbNrZyxVW7C.jpg)




