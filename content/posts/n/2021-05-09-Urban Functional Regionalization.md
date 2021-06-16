---
title: "2021-05-09: Urban Functional Regionalization"
date: 2021-05-09T19:45:40+08:00
categories:
  - 收藏
tags:
  - 札记
  - Regionalization
  - Urban functions
---
> Paper writing notes about urban functional regionalization.

<!--more-->


------------

<!-- content -->



### 札记

#### 概念/定义

- Regionalization
  - the task of partitioning a set of contiguous areas into spatial clusters or regions
  - a type of clustering method that defines spatially contiguous and homogeneous groups, also known as regions.
  -  partition the geographic space into a set of homogeneous and geographically contiguous regions
  - aptly described as ‘<u>spatially constrained clustering algorithms</u>’







### 2017-Creating multithemed ecological regions for macroscale ecology: Testing a flexible, repeatable, and accessible clustering method

- Ecology and Evolution (中科院3-4区)
- 逻辑清晰，写作蛮好的，值得借鉴
- regionalization的分析套路值得借鉴



#### Abstract

- **spatially constrained spectral clustering algorithm**: a spatially constrained spectral clustering algorithm that <u>balances geospatial homogeneity and region contiguity</u> to create ecological regions using multiple terrestrial, climatic, and freshwater geospatial data
- Identify **the most influential geospatial features**: identified which of the geospatial features were most influential in creating the resulting regions
- capture **regional variation**: tested the ability of these ecological regions to capture regional variation in water nutrients and clarity for ~6,000 lakes

#### Introduction

- Para. 1: 生态背景，看起来有点吃力
  - understand and predict ecosystem at broad spatial and temporal scales
  - translating fine‐scaled understanding to macroscales is difficult
  - spatial heterogeneity among ecosystems
- Para. 2:（生态背景下的）区域化方法介绍
  - a **regionalization framework** that <u>classifies the landscape into ecological regions</u>
  - under the **assumption** that <u>ecosystems within regions are more similar</u> (in properties and in responding to stressors) <u>than those across regions</u>
- Para. 3:目前研究方法潜在地限制当前的应用
  - **existing ecological regions** have characteristics that **potentially limit** their general application. 
  - limitations in the <u>availability of broad‐scale geospatial data</u>
  - <u>subjectively using paper maps</u>, leading to regions that <u>cannot be reproduced or easily modified for new purposes</u>
- Para. 4: 这一段的逻辑还是蛮清晰的，讲区域化方法还是广泛应用滴，但是呢有A和B两个局限，但是这两点又很重要，怎么办呢，那当然是研究一个方法可以解决这两个问题啦。
  - historic regionalization frameworks are <u>widely used</u> 
  - there have been <u>advances</u> in statistical and computational approaches for delineating objective and reproducible ecological regions
  - most of these newer methods <u>have not been broadly disseminated(传播) to or available</u> in a form easily adoptable by the ecological community.
  - many of these methods are optimized to maximize landscape homogeneity, <u>they do not always create contiguous regions</u>
  - **Region contiguity is useful for two important reasons**. 
    - First, such regions **help account for** broad‐scale **spatial autocorrelation** that is common among ecosystems. 
    - Second, contiguous regions are **useful for management** because they allow managers to **apply similar practices to nearby** but unstudied ecosystems.
  - Therefore, <u>we need methods that **create contiguous and homogeneous regions**, as well as **dissemination of these approaches** to the ecological community</u>.
- Para. 5: 这一段的写作逻辑真的太爱了！可以当写作模板！💛
  - 一句话陈述，我们做了啥：
    1. 应用一个新的方法：apply a newly published computer science clustering algorithm that creates customized ecological regions
    2. 看它能不能用：test its use for macrosystems ecology research
    3. 人人可用：make it available in an online repository
    4. 拔高：help fill the need for adaptable and flexible methods for creating regions
  - 关于空间限制算法的描述：a flexible method that allows users to **impose restrictions** on <u>whether spatially adjacent points should be in the same region</u>, thereby <u>influencing the clustering process</u> to create homogeneous regions that are also geographically connected (i.e., contiguous).
  - 论文里讲到，空间限制的谱聚类算法不是这篇文章提出的，但是我们扩展了他的应用：
    - This method was developed and tested using terrestrial landscape **data** for three U.S. states and was found to **outperform** three other algorithms for delineating ecological regions 
    - we expand on this previous work to:
      - 更大范围！—— create ecological regions **with a wider range** of nationally available data
      - 谁更重要！—— examine which of the 52 geospatial features were most **influential** in creating these regions to determine how important individual geophysical features are for ecological region delineation
      - 空间差异！—— test the ability of the resulting 100 ecological regions to capture regional variation in lake characteristics that were not used to develop the regions
  - 人人可用：<u>We make this algorithm freely available with an accessible user interface for other researchers to use and modify, including the ability to</u>: create different numbers/sizes of regions; use a subset of themes or different combinations of measures of the terrestrial, atmospheric, and freshwater landscapes; and create regions for a different spatial extent (e.g., state, nation, and continent).
  - 拔高：This objective and reproducible method and available code for creating ecological regions are designed to <u>support a wide range of</u> macroscale ecology applications.

#### Materials and Methods

- 研究单元——base geographic unit:
  - the U.S. Geological Survey 12‐digit <u>hydrologic(水文的) unit</u> (HU‐12), which is based on river basins
  - There are **20,257** HU‐12s in the study extent, ranging in land <u>area from 0.35 to 1,276</u> km2
- 单元特征——natural geographic variables:
  - 52
  - grouped into <u>three themes</u>: terrestrial landscape features, climate features, and freshwater landscape features
- LAGOS‐NE dataset:
  - includes <u>lake‐specific water quality and chemistry data</u> compiled from 54 individual datasets for a subset of ~10,000 lakes in the study extent
  - for independently <u>testing the ecological regions</u> created in this study
- Schematic illustrating the procedure for creating ecological regions:![](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513103705.jpeg;%20charset=UTF-8)
  1. 预处理过程——Data preprocessing:![image-20210513111333997](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513111334.png)

     - **remove** HU‐12s that included spatially isolated landscape features (e.g., islands and peninsulas)
     - **fill in missing values** in the geospatial database through interpolation
     - remove egregious outliers
     - PCA scores: 降维
  2. two matrixs and joint similarity matrix:

     - feature similarity matrix: a landscape feature similarity matrix that <u>measures landscape homogeneity</u> was computed using the <u>Gaussian radial basis function</u>
     - spatial constraint matrix: a <u>binary‐valued</u> spatial constraint matrix was constructed based on HU‐12 contiguity (i.e., <u>1 if the HU‐12s share a border; 0 if the HU‐12s do not share a border</u>). The spatial constraint matrix is used to <u>guide the clustering process into finding spatially contiguous regions</u>.![image-20210513105443622](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513105443.png)
     - Hadamard product:![image-20210513105724718](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513105724.png)
  3. spectral clustering algorithm:![image-20210513110218622](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513110218.png)
  4. two metrics for evaluating a regionalization framework:![image-20210513110643836](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513110643.png)

     - **within-cluster sum-of-square error** (SSW) to quantify the landscape homogeneity within the regions, A **lower** SSW implies higher homogeneity of landscape features within regions
     - percentage of spatial constraints preserved by the clustering algorithm(PctML), a measure of cluster contiguity. This metric ranges from 0 to 1 and the **higher** the metric, the more spatially contiguous were the resulting regions.
     - created nine sets of ecological regions:![image-20210513115623061](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513115623.png)
  5. **number of regions**:
     - 聚类过程中确定聚类数的传统方法——standard approach to choose the optimal number of clusters
       - plot values of an internal cluster validity index such as SSW against the number of regions and <u>identify the inflection point in the monotonically decreasing curve</u> (识别单调递减曲线的拐点)
       - 存在的不足：
         - this approach is subjective and the inflection point may not always be easily identified
         - it does not consider the statistical significance of the regions compared to purely random clustering (i.e., no consideration of landscape homogeneity or region contiguity). 
         - Worse still, the monotonically decreasing relationship between SSW and number of regions is observed even for purely random clustering (i.e., no consideration of landscape homogeneity or region contiguity).
     - 文中优化的方法：compared the SSW of the regions created with spatially constrained spectral clustering <u>against the average SSW for 200 randomly created sets of regions</u> to ensure that the improvement in SSW as the number of regions increases was <u>statistically significant</u>
       - computed **the ratio of slopes** for the two approaches as the number of regions increases
       - If the constrained spectral clustering approach provides **little improvement** in SSW compared to the random clustering approach, then **this ratio approaches 1** on plots of empirical curves and <u>indicates an optimal number of regions</u>.
       - the number of regions increased from **5 to 1,000**（with a step size of 5 from 5 to 600 clusters, a step size of 10 from 610 to 800 clusters, and a step size of 50 from 850 to 1,000 clusters）
  6. Determining drivers of ecological regions:![image-20210513115952196](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513115952.png)

     - **evaluated the relative importance** of the 52 geospatial variables for region formation using a random forest algorithm
     - The OOB error estimate
     - Gini impurity criterion
  7. Testing the ability of ecological regions to capture regional variation:![image-20210513120230984](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210513120231.png)
     - examining SSW for the two lake characteristics
     - examined the ratio of SSW:SSB in order to compare relative amounts of within‐ and among‐region heterogeneity across response variables and ecological regions



### 2020-Eﬃcient regionalization for spatially explicit neighborhood delineation

- IJGIS: https://doi.org/10.1080/13658816.2020.1759806
- 生词多的我都觉得自己不是GIS 这个专业的了。



#### Abstract

- Neighborhood delineation ➡️ <u>identify the most appropriate spatial unit</u> in urban social science reasearch
- the true number of neighborhoods (k parameter)
- Existing approaches
  - pre-speciﬁcation of a k-parameter
  - either nonspatial or lead to noncontiguous or overlapping regions
- In this paper
  - propose <u>the use of max-p-regions</u> for neighborhood delineation: the geographic space can be <u>partitioned</u> into a set of <u>homogeneous and geographically contiguous neighborhoods</u>
  - computational challenges for large-scale neighborhood delineation



#### Introduction

- Para. 1:
  - An increasingly important technique in the ﬁeld of GIScience: <u>the identiﬁcation of distinct sub-regions or neighborhoods within a study area using unsupervised learning methods</u>
  - generally <u>categorized</u> as <u>regionalization methods</u>
  - aim to <u>partition the geographic space into a set of homogeneous and geographically contiguous regions</u>
  - regionalization algorithms might be more aptly described as ‘<u>spatially constrained clustering algorithms</u>’
- Para. 2:
  - One important application of spatially constrained clustering
  - the total number, spatial conﬁguration, and internal composition of neighborhoods are all unknown a priori （邻里的总数、空间结构和内部组成都是先验的）.
- Para. 3:
  - The ﬁrst application: regionalization is leveraged method of <u>data processing</u> used to develop new <u>primitive spatial units</u>(基础的空间单元) that <u>have better statistical reliability</u>
  - Spielman and Singleton (2015): social surveys (e.g. the census)
- Para. 4:
  - The second application: regionalization is used to <u>identify unique and discrete social neighborhoods</u> according to their <u>demographic composition</u>
  - Rey et al. (2011): geodemographic analysis, examine the dynamic footprint of social neighborhoods over time
- Para. 5:
  - there is a clear need for exploration and development of novel approaches to regionalization that are <u>scalable, eﬃcient, and able to ingest vast amounts of data in short cycles</u>.
  - max-p-regions
  - proposed a new eﬃcient algorithm to address the computational challenges

### 2021-A quantitative comparison of regionalization methods

- IJGIS: https://doi.org/10.1080/13658816.2021.1905819
- 分区方法比较全面的一套综述
- 这是我见过IJGIS最长的Introduction了

#### Abstract

- Regionalization: the task of partitioning a set of contiguous areas into spatial clusters or regions
- yet <u>few quantitative comparisons</u> have been conducted
- the number of regions
- the simulated benchmark data set
- Model families are defined with respect to <u>regions’ shapes, value-mixing between regions, and the number of underlying spatial clusters</u>
- internal and external measures of regionalization quality
- investigate the computational efficiency
- implications on defining ecological regions



#### Introduction

- Para. 1: 主要讲分区的意义和应用
  - Datadriven regions
    - public health
      - Applications:
        - spatial groups of disease incidence
        - epidemiological analysis (流行病学分析)
      - Significances:
        - classifying disease prevalence
        - delineate areas prone to a specific disease type
    - regions on social networks
      - defined based on the similarity of people’s interests to define communities
    - location-allocation problems
      - Applications
        - renewable source optimization to serve cities’ energy demands
        - environmental planning
      - Significances
        - optimize energy consumption and data routing
    - regions for dynamic systems
      - delineate climate zones
    - define socio-economic regions
    - ecological regions （生态分区）
    - electoral and school districting
- Para. 2: 分区问题的数学定义
  - solve a constrained optimization problem  to **maximize within-region similarity and between-region dissimilarity** under spatially derived constraints
  - n spatial units; vector xi with p variables; partition the data into k regions;
  - distance measure
  - the spatial constraint defined by geographical adjacency.
- Para. 3: Regionalization方法分类
  - spatially implicit models
    - generally based on traditional and non-spatial clustering methods producing a <u>first</u> solution that is <u>afterwards</u> updated by enforcing the spatial constraints
    - force the regions to be as homogeneous as possible, but the spatial contiguity is not always guaranteed
    - an ad hoc <u>post-processing step</u>
  - spatially explicit models
    - models spatial constraints <u>explicitly</u> to ensure spatial contiguity within the resulting regions
    - grouped as exact, heuristic, and mixed-heuristic models
      - Exact models search for the optimal regions among <u>all</u> possible regionalizations
      - Heuristic models constrain the search space for the optimal solution to find regions <u>efficiently</u>
      - mixed-heuristic models aim to <u>combine</u> heuristic models’ computational efficiency with exact models’ compactness
    - Others: 
      - segmentation algorithms
      - graph-based partitioning algorithms
- Para. 4&5:一些其他的方法，本文不赘述
  - seeded region growing (SRG) algorithms
  - automated zoning procedure (AZP)
  - spatially explicit regionalization methods: finding underlying regions with constraints related to regions <u>without the definition of a target number of regions</u>
  - Spatially implicit regionalization methods <u>suffer from subjectivity due to the ad-hoc post processing</u> required to ensure spatial contiguity
- Para. 6:显式分区方法的缺陷
  - spatially explicit regionalization methods:
    - exact methods are computationally intensive
    - heuristic models: 
      - their solution’s optimality is not guaranteed
      - constrained search space can result in heterogeneous regions where spatial units in the same region are significantly dissimilar
- Para. 7&8&9: In this paper:
  - present a quantitative review
  - expand and complement a recent paper
  - conduct a rigorous statistical analysis of regionalization algorithms
  - analyze the computational performance
  - compare the performance of the state-of-the-art regionalization methods with SKATER
  - evaluate algorithm performance quantitatively
    - degrees of separability between the spatial clusters
    - number of underlying regions
    - realizations for a given scenario
  - define data-driven ecoregions: 
    - consider the utility of different regionalization algorithms on regionbuilding to define ecoregions
    - compare our results to a widely known, expert-defined regionalization of ecoregions
- Para. 10: Contributions
  - comparative performance of a large set of regionalization methods
  - Demonstrate results from a large set of supervised and unsupervised regionalization quality metrics
  - Perform sensitivity analysis to evaluate the algorithm performance for different scenarios
  - Evaluate the performance of the different algorithms for an increasing number of spatial units
  - Discuss the implications of performance disparities from the synthetic study



#### Methodology

- unsupervised evaluation metrics:
  - Calinski-Harabasz index
    - quantifies the <u>value-based compactness</u> (紧凑度) of k regions by comparing the average of within and between region sum of squares
    - The Calinski-Harabasz index will be **high** for a region map containing distinct (dissimilar) regions that consist of similar spatial units.
  - Sum-Squared Errors:![image-20210514110845211](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210514110845.png)

- **SKATER**
  - it **excels** across most of our evaluation criteria when compared to the other regionalization techniques
  - for cases where data do not possess well-defined regions, <u>tree-based methodologies such as SKATER can define the regions with the most homogeneity</u>.





### 2019-A regionalization method for clustering and partitioning based on trajectories from NLP perspective

- IJGIS:https://doi.org/10.1080/13658816.2019.1643025
- 手机信令轨迹+Word2Vec
- 最开始看的，关于分区的几篇文章之一

#### Abstract

- a novel regionalization method to <u>cluster similar areal units and visualize the spatial structure</u> by considering all trajectories in an area into a word embedding model.
- the result depicts the <u>underlying socio-economic structure</u> at multiple spatial scales
- evaluate its performance by <u>predicting the next location of an individual’s trajectory</u>



### 2019-Hierarchical community detection and functional area identification with OSM roads and complex graph theory

- IJGIS: https://doi.org/10.1080/13658816.2019.1584806
- 关于社区检测，我的“入坑”读物

#### Abstract

- urban road network structure ➡️ understanding the distribution of urban functional area
- communities of urban road roads
- Infomap commnity detection algorithm
- results:
  - the distribution of communities at different levels
  - explored the functional area characteristics at the communities scale
  - can be used as a basic unit

![image-20210514153720812](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210514153720.png)



### 2021-Partitioning urban road network based on travel speed correlation

- International Journal of Transportation Science and Technology: https://doi.org/10.1016/j.ijtst.2021.01.002
- 了解学习下交通领域背景
- 标准的abstract模板

#### Abstract

- urban trafﬁc management
- urban road network partition
- existing study
  - the spatial relationship of road sections are introduced
  - fails to capture the travel speed correlation between road sections with far distance
- this paper
  - travel speed correlation between road sections
  - <u>fast unfolding method</u> is used to divide urban road network into sub-partitions of densely correlated road sections
  - A case study is conducted by using <u>taxi GPS dataset in Shanghai</u>
- Results:
  - the travel speed will generate high correlation even if the road sections are not spatially connected or close
  - divides the road network in Shanghai into <u>77 sub-partitions</u> with strong intro-correlation of travel speed pattern
  - Comparing the result with Ncut algorithm with different spatial constraints, <u>generate evenly distributed and spatially compact sub-partitions</u>.

#### Introduction

- Para. 1: 交通分区的意义
  - **increasingly serious problems in cities**: urban congestion, extra gas emissions and low transportation efﬁciency
  - Urban road network partitioning is **a fundamental step** in traffic management, control, simulation and policymaking.
    - the citywide management strategy is <u>hierarchical and regionalized</u>.
    - ensuring that the road sections with a similar travel speed pattern are in the same sub-partition
    - <u>applying the same controlling strategy</u> is the ﬁrst step to guarantee the effectiveness of the overall controlling system
- Para. 2&3&4&5: 面临挑战——facing the following challenges
  - the ﬁrst challenge is <u>how the partition method considers the urban trafﬁc characteristic</u>
  - the second challenge is how the partition method divides the road network so that <u>a network partition can be spatially compact or continuous</u>
    - the most existing road network partitioning approaches <u>heavily depend on topological relationships of road segments</u>
    - when spatial constraints are imposed on the relationship between road sections, <u>the relationship between non-spatial continuous or adjacent road sections will inevitably be ignored.</u>
  - <u>the travel speed of road sections highly related</u> to each other
- Para. 6:
  - A **methodology framework**:
  - capture and quantitatively express the travel speed correlation
  - the road network partition methodology proposed is based on trafﬁc ﬂow characteristics
  - handle a citywide road network and generates the road network partition with a short computing time



#### Others:

- potential control strategies to alleviate trafﬁc congestion <u>should be designed based on the sub-partition of the urban road network</u>
  - if a cluster contains subregions with signiﬁcantly different levels of congestion, the control strategies will be inefﬁcient
  - road network partitioning is the ﬁrst step to <u>space-parallel distributed transportation simulation</u>
- the principle of road network partitioning:
  - Non-overlapping
  - Travel characteristic based
  - Spatially compact
  - balanced size
- Framework:![image-20210517160045923](https://cdn.jsdelivr.net/gh/xunhs/image_host@master/PicX/20210517160045.png)

---

<!-- pic -->
![anh-tr-n-jr6oNjP75Y8-unsplash.jpg](https://img.maocdn.cn/img/2021/05/09/anh-tr-n-jr6oNjP75Y8-unsplash.jpg)