---
title: Papers Reading-Taxi Trajectory/GPS Records
date: 2020-02-24T09:14:16+00:00
categories:
  - 收藏
tags:
  - 出租车轨迹
  - 轨迹
  - Trajectory
  - 论文
  - 论文阅读
  - Human activity data
  - NLP
  - semantic
  - geo-semantic mining
  - geo-semantic
  - spatial context
  - interactions
  - vehicle moving paths
---


> 论文阅读，感兴趣点、关键词整理

<!--more-->


#### 博士论文: 基于出租车数据的城市居民活动空间与网络时空特性研究
- 大多数研究对于活动数据仅使用活动开始和结束的位置点数据，`忽略行驶轨迹，数据利用不充分`。
- 从个人轨迹段和POI时空吸引力棱镜的时空关系出发，`确定个体活动所在的POI。`
- 空间句法（space syntax）被用来解释道路网结构与城市布局之间的关系
- 空间句法将现实空间抽象表达为符号空间，如将道路段抽象为点，并利用句法模型的计算与分析将具有拓扑关系的图解与变量一一对应，成功`将城市空间引入定量的表达`。
- 基于重力模型计算可达性指标的方法
- 将空间划分为格网，定义出租车GPS轨迹为伪格网序列(pseudo cells)
	- 计算轨迹长度并不是出租车实际行驶的网络距离
	- 忽略路网环境导致在分析轨迹特征时无视城市交通条件
- 判别轨迹异常程度的指标
	- 距离约束：实际距离与最短路径之间的量化差异
	- 规避路段约束
	- 时间约束。
- 出租车经过的城市关键结点

#### Journal Article: 2013-Land-Use Classification Using Taxi GPS Traces  
- Refer: [IEEE Transactions on Intelligent Transportation Systems](https://ieeexplore.ieee.org/document/6266748)
- **AAAA**
- GPS traces of vehicles
- human mobility and activity information, which can be closely related to the land use of a region
- recognizing the social function of urban land 
- pick-up/set-down(set-down表示下车点比较少见，`用drop-off吧`) dynamics/pattern:上下车模式
- 作者总结的贡献：  
    1. 关于Remote sensing based land use classification
        - Previous (2013) land-use classification research was based on the physical properties of studied objects in remote-sensing data
        - Most urban land-use classification research has used remote-sensing data, particularly satellite images.
            - satellite resolution
            - spectral reflectance and the nature of the materials
            - methods: Pixel-based classification & Object-based classification
    2. we verify that the social function of a certain urban area can be characterized by `the temporal and spatial dynamics` of the taxi pick-up/set-down number (`the temporal and spatial dynamics: simply describe the variation of pick-up/set-down number over time`)
        - 换一种说法:  verified that there is an inherent relationship between land-use classes and the temporal pattern of taxi pick-up/set-down dynamics
    
    3. designed `six features` extracted from the pick-up/set-down data of different time lengths  
        - six features(P7):![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200413145419.png)![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200413145500.png)
        - a recognition accuracy of 95% (太高了？几分类？)`534 regions with eigth kinds of social functions`:![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200413145720.png)
        - Four classical classifiers are evaluated(All the parameters for the algorithms are optimized.):
            - linear-kernel SVM (Best Classification Result with Feature I + II)
            - k-nearest neighbor
            - linear discriminate analysis
            - three-layer BP
    4. social function transition (变更) of regions
- 相关文献
    1. taxi trace data
        - `Ubiquitous mobility data` contain information that is important for the smart environment; reflect urban traffic behaviors and convey lots of information about a city
        - Taxi trace data could be used for:![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200413134514.png)
    2. urban land-use classification
    3. approaches using mobility data for land-use classification in the literature
        - clustering algorithm 
        - more land-use classes
        - origin-destination(OD) flows

#### Journal Article: 2020-The Traj2Vec model to quantify residents’ spatial trajectories and estimate the proportions of urban land-use types 
- Refer: [IJGIS](https://doi.org/10.1080/13658816.2020.1726923); [iGEODATA](https://mp.weixin.qq.com/s/Io7i4iiOZoxaJuwqwJlDZg)
- **AAAAA**
- Abstract
    1. geo-semantic mining approach: quantify the trajectories of residents as highdimensional semantic vectors
    3.  RF: model the relationship between the semantic vectors and
mixed urban land uses
    4. <u>讲的一手好故事</u>[`分析混合指数和旅行距离之间的关系，发现他们之间有一种显著的（弱的）负相关关系，表明土地利用混合指数增加，居民出行距离缩短，进一步减少了能源消耗。`]: analyzing the mixing index and the travel distance, a weak but significant negative correlation between them; an increase in the degree of mixing will reduce the travel distances of residents. 
- Introduction
    - urban land use 
        1. the spatial distribution of urban land use 
        2. urban land uses and urban spatial structures have become increasingly `diverse and sophisticated`
        3. obtaining `qualitative and quantitative` data on mixed urban land uses `quickly and accurately` is very important for understanding and managing cities
        4.  There is a `general lack` of such studies, because mixed urban land use is difficult to estimate using `conventional methods`.
    - trajectory data (套路讲的是极好的)
        1. [所谓居民活动数据]`Human activity data` can provide more detailed and accurate information for analyzing mixed urban land-uses since urban land use is defined as `the use of the urban space by residents and the activities within that area` = (the data contain valuable information on how people utilize urban spaces)
        2. [LBSs能够提供居民活动信息，并用于城市空间结构研究]The rapid development of location-based services (LBSs) provides us with a large amount of `human activity information` that can be used to measure urban spatial structures and land uses. In particular, trajectory information is generated by residents in their daily lives and can `represent the resident’s behavioral purposes`.
        3. [以往的研究仅使用简单的特征] previous studies have only taken some simple features from human activity data, such as the `frequency and volume`. These methods may `waste the majority of the spatial information and the inner spatial correlations` in human activity data
    - `geo-semantic mining techniques`(大论文可参考)
        1. explore the spatial semantic features of geospatial data
        2. [NLP中的语义挖掘]In NLP, semantic mining refers to the `transformation` of words, phrases, signs, and symbols into forms that `computers can recognize and understand the relationships` between them. Semantic information is a digital high-dimensional feature vector that can fully characterize these relationships.
        3. [地理语义挖掘解释]geo-semantic mining refers to `mining potential relationships in geographical data`. By exploiting the potential relationships, we can fully extract the information inside geographical data and apply it in various geographic applications.
        4. [目前的地理语义挖掘研究]Existing research on geo-semantic mining show that the semantic model can well discover the potential semantic information of the geospatial data, and the obtained information can be used to quantify the relationship between the urban land uses and geospatial data
        5. [文中的语义与传统的地理信息语义的关系]: In particular, there are some differences between the semantics here and the traditional geographical semantics. `The semantics here is an abstract concept that uses feature vectors to represent the potential relationship within geographic data.` Traditional geographic semantics refers to describing the meaning of spatial data and the relationship between them, and making the semantics of geographic information explicit.
    - consider the `spatial context` in spatial data
        1. introduced the `Word2Vec` model to measure the potential contextual relationships between POIs and obtained satisfactory results in the classification of detailed urban land uses   
        2. [POI数据是空间离散的，构建连续序列的方法很大程度上影响空间上下文关系]POIs are `spatially discrete`, so some methods were also developed to construct a continuous dataset from POIs. The method used to construct a continuous POI dataset largely `affects the spatial contextual relationships`, thus affecting the result of urban land-use identification
        3. [由POI的空间离散引申到轨迹的连续]A trajectory is `continuous in space`, and a person’s travel information can reflect the use of urban space. Therefore, `it is expected that` the use of the Word2Vec model to explore the potential semantic information in a trajectory can help us to better understand structures and land uses.

#### Journal Article: 2017-Road2Vec: Measuring Traffic Interactions in Urban Road System from Massive Travel Routes
- Refer: [IJGI](https://doi.org/10.3390/ijgi6110321)
- **AAA**
- Abstract
    -  [交通互作用？(反复出现)/我觉得用<u>Traffic-Flow Dependency</u>更恰当一些]`traffic interactions` among urban roads; quantify the implicit traffic interactions; can be effectively utilized for quantifying complex traffic interactions among roads and capturing underlying heterogeneous and non-linear properties
    -  large-scale taxi operating route data
- Introduction
    1. [城市道路的交通状况通常受邻近道路的影响]The traffic states of urban roads are often influenced by their neighboring roads. Different terms, such as `spatial dependency/relationship in traffic and spatial correlation`, are used in the literature to express such relationship between neighboring roads. In this paper, we use the term traffic interaction to describe the traffic influence between neighboring roads, which is fundamentally caused by the `dynamic vehicle movements` from one road to another.
    2. [道路之间的交互源于车辆的移动；因此道路之间的本质联系可以从车辆轨迹之间获取]Essentially, the traffic influence among roads originates from numerous vehicle movements on road systems; hence, the inherent relationships among roads should be extracted from massive vehicle travel routes. 
    3. [向量之间的相似性解释]According to the principle of word embedding models, a `high similarity` between two word vectors indicates that `the two words co-occur frequently in textual documents or their local contexts are very similar`. Correspondingly, `high similarity between two road segment vectors` indicates that two road segments frequently co-occur in travel routes or they frequently share common upstream and/or downstream segments in travel routes. Both situations indicate that there are strong traffic interactions.
    4. [浮动车数据] (low-frequency) floating car data (FCD) collected by GPS-equipped taxies; [乘客上下车位置]passengers’ pick-up and drop-off locations; mapping each GPS point to a road segment=>travel routes
    5. [看来大家都喜欢计算这个平均相似度]calculate the `average similarities` of vectors among the first-order, second-order, third-order, and fourth-order neighboring roads, respectively

#### Journal Article: 2019-Identifying spatial interaction patterns of vehicle movements on urban road networks by topic modelling  
- Refer: [CEUS](10.1016/j.compenvurbsys.2018.12.001); [未名时空](https://mp.weixin.qq.com/s/-OaerajZqJo_u2mAg92GCg)
- **AAA**
- 与上一篇(Road2Vec)同作者`traffic interaction` => `spatial interaction`
- Abstract
    1. investigate the `spatial interactions` derived from human movements = identify spatial interaction patterns of vehicle movements on urban road network
    2. [城市居民移动受限于车辆和城市路网=><u>道路之间的交互</u>]in most cases, human movements are carried by vehicles and constrained by the underlying road network, which causes the `interactions among roads`
    3. "strokes" (i.e., `natural streets`) are chosen as geographical units to represent the `vehicle moving paths`.


#### Journal Article: 2017-Street as a big geo-data assembly and analysis unit in urban studies: A case study using Beijing taxi data
- Refer: [Applied Geography](https://www.sciencedirect.com/science/article/abs/pii/S0143622816301734)
- **AAAAA**
- 学会<u>戴帽子</u>
- Abstract
    - understanding urban environments
    - Spatial assembly: an essential analytical step to `summarize and perceive geographical environment from individual behaviours`
    - [街道尺度]the adopted spatial units for data aggregation remain areal in nature; sensing cities from a street perspective, emphasizes the significance of street units in quantitative urban studies
    - three-month taxi trajectory dataset
    - [<u>道路的动态功能和承载力</u>]explore the spatio-temporal patterns of urban mobility on streets, `cluster streets into nine types` based on their `dynamic functions and capacities`
    - is able to effectively `minify` the modifiable areal unit problem (MAUP)
    - [意义套装]sense urban dynamics, depict urban functions, and understand urban structures
- Introduction
    1. [传感器数据]Through `automated and routine movement tracking` of individuals, `various forms of locator devices work as sensors` to collect geospatial data and `characterize the activity of a city` in both spatial and temporal perspectives
    2. [两个角度的帽子]From the perspective of individuals, citizens play the role of voluntary sensors and produce plenty of volunteered geographic information. At the collective level, the distribution of geographic phenomena such as `land use (or social function) and the pattern of spatial interaction flows` can be investigated after spatio-temporal aggregation of individual behaviour data.
    3. [两种属性用于理解城市问题]Utilizing massive amounts of geospatial data, `the first-order distribution of urban attributes (e.g., economic indices, population intensity, condition of public facilities)`, as well as `second-order interactions (e.g., human movements, flow of goods, financial flows, social ties)` can be used to better understand human mobility, urban functions, and urban structures
    4. [空间集配(?)中的MAUP问题]Spatial assembly: inevitable to confront the issue of spatial resolution (or scale) when mapping individual details onto regular or irregular units
        - Voronoi polygons
        - regular grids
    5. [街道尺度研究的意义]using streets as the basic elements to characterize urban functions and understand urban structures.
        -  the street system is never an insignificant part of a city. Lynch的城市映象,path为首
        -  [城市内部的移动受路网/街道限制]It is now generally accepted that the physical movement in an urban space is usually constrained by a road network and streets interlink urban functions physically and cognitively
        - [面状研究单元的替代]street unit is `a promising substitute for areal units` and can help us uncover hidden knowledge concealed under areas
- Methodology
    1. Temporal patterns of pick-ups and drop-offs
    2. Association of `street classifications` with dynamic street
patterns
        1. Hierarchical clustering based on dynamic street functions and
capacities: (1) vector; (2) normalization; (3) unsupervised hierarchical bisecting k-means clustering.
        2. Characterizing street types by dynamic functions and capacities: (1) classified into nine types.
        3. Uncovering urban structures in the street perspective: (1) Uncovering urban structures in the street perspective; (2) detect communities with the best modularity. 
    3. The complexity of streets
        1. [未考虑土地利用的复杂性]comprise the complex land uses along streets
        2. [未考虑双向道路属性]the `bidirectional nature` of streets is not considered in our work for simplicity. 


#### Journal Article: 2016-Incorporating spatial interaction patterns in classifying and understanding urban land use
- Refer: [IJGIS](https://www.tandfonline.com/doi/full/10.1080/13658816.2015.1086923)
- Abstract
    - Land use classification
    - travel behaviour
    - [(大数据)作为传统遥感影响数据方法的一种补充]complementing the outcome of traditional remote sensing methods
    - spatial interaction patterns
    - [未能得到验证和分析]have rarely been examined and analysed
    - unsupervised land use classification method
- Introduction
    1. Traditionally, researchers collect residents’ trip information by travel surveys
    2. understand human movements and urban built environments
    3. [城市居民活动的追踪/足迹]the spatial footprints of citizens’ activities
    4. [人们的移动是可以预测的]people’s mobilities are highly predictable
    5. emphasizes the `social function` of a place
    6. [相同土地利用类型的区域用相似的时空活动模式]The routines of people guarantee that places of the same land use type, to some extent, `share similar temporal activity variations`.
    7. [仅仅扩充时空活动变量是无法补救的]simply adding more features of temporal activity variations for land use classification is not a remedy. 
    8. [考虑轨迹连续性，提取交通流(特征)]consider the movement between two consecutive activities as a travel and extract traffic flows from the trajectory data
    

#### Journal Article: 2019-Detecting regional dominant movement patterns in trajectory data with a convolutional neural network  

- Refer: [IJGIS](https://dx.doi.org/10.1080/13658816.2019.1700510)
- Abstract
  - movement pattern detection  
  - detect regional dominant movement patterns (RDMP) in trajectory data
  - a novel feature descriptor
    - directional flow image (DFI)
    - to store the local directional movement information 
  - a classification model
    - TRNet, designed based on CNN
    - trained with a synthetic trajectory dataset [合成轨迹数据]
  - a sliding window detector - detect RDMP at multiple scales 
  - a clustering-based merging method - prune the redundant detection results
  - Evalution
    - high training accuracy
    - experiments on a real-world taxi trajectory dataset
  - Introduction
    - [移动模式重要性] **Movement patterns embedded in trajectory data** can provide valuable information for the tracked objects and the context, which play an important role in many applications.  
    - previous work  => low generalization capability  
    - [深度学习应用在交通领域的一些尝试] several attempts have been made in the transportation domain to employ deep learning methods to exploit the value of big data.  
    - Conventional CNN models 
      - the input of CNN is required to be a **fixed** tensor  
      - A trajectory can contain a **variable** number of points, which belongs to vector data  
      - a trajectory provides two additional pieces of information: **direction and connectivity** between points  
    - To address the above problem,  this paper...

***
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/spring/20200229205044.jpg)