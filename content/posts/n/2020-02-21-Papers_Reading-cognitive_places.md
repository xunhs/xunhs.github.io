---
layout: post
cid: 107
title: Papers Reading-Cognitive Places&Urban Perception
slug: 107
date: 2020-02-21T09:12:41+00:00
updated: '2020/02/23 10:27:55'
status: publish
author: Ethan
categories:
  - 收藏
tags:
  - 论文
  - cognitive places
  - street view images
  - POIs
abbrlink: '7591e184'
---

> papers about investigating cognitive places&urban perception
<!--more-->



### 相关概念

<u>场所(place)</u>是被赋予了个体经验、生活与情感意义的空间位置或区域，是理解地理环境的重要途径之一。（大众点评数据下的城市场所范围感知方法）

<u>Cognitive region</u> boundaries are typically substantially <u>vague</u> and their membership functions are substantially <u>variable</u> – the transition from outside to inside the region is imprecise or vague, and different places within the region are not equally strong or clear as exemplars of the region. (Vague cognitive regions in geography and geographic information science)

vague geographic regions  

taxonomy of geographic regions  



### Regions in geography: Process and content

Montello-2003-[Foundations of geographic information science](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Regions+in+geography%3A+Process+and+content&btnG=)

#### Introduction

- regions
  - <u>the concept of regions</u> has nearly always been of central importance    
  - <u>the identification, description, and explanation of regions</u>  has played a critical role  
- in this eassy
  - revisit the regional concept
    - place can be considered a subset of the regions
  - contrast the concept of geographic regions  
  - propose a taxonomy of geographic regions: administrative, thematic, functional, and cognitive regions  
  - differentiate formal regions into administrative and thematic regions  
  - consider the individual vs. social nature of cognitive regions  
  - consider the important issue of boundaries  
    - consider the important issue of boundaries  
    - how it applies to the four region types  
  - The special status of administrative regions   

##### What are geographic regions  

- P1: Regionalization=>Categorization=>Categories=>Spatial categories—regions  
- P2: Geographic regions have certain shared properties    
- P3: Geographic regions are thus examples of spatial regions in general  
- P4: Geographic regions need not be contiguous  

##### Regions in thought  

#### PROCESS- AND CONTENT-BASED TAXONOMY OF REGIONS  

##### new taxonomy  of regions

- **Administrative regions** are formed by legal or political action, by decree or negotiation.  
- **Thematic regions** are formed by the measurement and mapping of one or more observable content variables or themes  
- **Functional regions** are formed by patterns of interaction among separate locations on the earth  
- **Cognitive regions** are produced by people’s informal perceptions and conceptions  

##### traditional region taxonomy

- formal, functional, and general regions  
- internal similarity and external dissimilarity of regions  

##### comment  

In applying the taxonomy, it is critical to recognize that <u>people use the same region label at different times to refer to different regions</u>; they also use it to refer to different types of regions. ( 在应用分类法时，关键是要认识到，人们在不同的时间使用同一个区域标签来指代不同的区域；他们也用它来指代不同类型的区域。)



### Vague cognitive regions in geography and geographic information science

Montello-2014-[IJGIS](https://doi.org/10.1080/13658816.2014.900178)

- regions are **<u>spatial categories</u>** (空间范畴):  A region encompasses places that are internally similar to each other and externally dissimilar to places outside the region.
- - A <u>region</u> encompasses <u>places</u>
  - internally <u>similar</u>; <u>dissimilar</u> to places outside the region
- All regions have <u>boundaries</u>
  - One of the <u>most important properties of boundaries</u> is that they <u>vary</u> in their precision or sharpness or, conversely, their *vagueness*.
  - <u>geographic boundaries</u> are not sharp at all but are really two-dimensional features – regions – themselves
  - <u>Vague boundaries</u> are transition zones (过渡区域) rather than lines between neighboring regions, but they are just as real as sharp boundaries
- Reseach about boundaries
  - discuss different reasons for boundary vagueness
  - quantifying and representing vague boundaries in computational systems / cartographic depiction
  - explored fuzzy logic as a formalization of vague boundaries
- Taxonomy of geographic regions
  - <u>administrative, thematic, functional, and cognitive regions</u>
  - Cognitive regions (traditionally called ‘perceptual’)
    - <u>regions in the mind</u>, reflecting informal ways that people organize places
    - can be idiosyncratic to a single person but are often shared among members of cultural groups (因人而异, 同时具有群体一致性)
    - reflect the type of <u>spatially categorical thinking</u> that so highly characterizes human thought and communication
    - substantially <u>vague boundaries</u>: The transition (转变) from inside to outside the cognitive region is usually a probabilistically graded zone of significant width
    -  <u>variable membership functions</u>: As a corollary (必然结果) to boundary vagueness, their membership functions are variable or probabilistically graded so that all places within the region are not equally strong or clear as members or exemplars of the region.



### Investigating urban metro stations as cognitive places in cities using points of interest  

- extracting and understanding the cognitive regions 
	- extract the cognitive regions of metro stations
	- identify the semantics of metro stations
- polygon generation techniques
- detect the place characteristics of urban metro stations
- urban metro stations are **typical cognitive places** <u>perceived by the crowd</u> through interacting with the surrounding society and environment,  which are characterized by <u>**vague boundaries**</u> and <u>**rich semantics**</u>（定义类）
- identify the semantics of the regions that can **reflect the crowd's impressions and perceptions**. (意义类)
- geotagged data are not ideal for studying the metro station areas **owing to issues of completeness and biases**（复杂性和有偏性）
- **Assuming** that frequently co-occurred place names on web pages **implies a strong relatedness between them**, researchers have investigated the relationships between geographical entities（假设）


### Representing place locales using scene elements
- locale indicates the **physical settings where everyday-life activities take place**, including visible and tangible aspects of a place such as buildings, streets, parks, etc. （定义类）
- sense of place refers to the **human experience and nebulous meanings** associated with a place（定义类）
- a vague cognitive region of a place, **which mines place semantics regarding human activities and perceptions** 
- how to **formalize the concept of place** with respect to locale and how to build a quantitative representation of locale remain unclear. 
- analyze the **physical appearance** of an urban space by photos（物理视觉）
- enabled by the proliferation of **computer vision and deep learning techniques**, it has been proven possible to acquire the semantic information of every single pixel in a natural image with high accuracy, thus improving our ability to **semantically understand scene content**（计算机视觉）
- the purpose of this study is to **formalize the concept of place** in terms of locale - the physical appearance of place
- employ an **image semantic segmentation technique** to parse street-level images and obtain 64 scene elements (building, sky, grass, etc.) that constitute a typical street scene（图片语义分割技术）
- The scene visual descriptor enables the carrying out of measurements among places and **contributes to the calculations between place and other spatial, demographic, and socioeconomic factors**（贡献类）
- The street scene ontology **illustrates the semantic relationships** among a certain number of scene elements to **support qualitative analysis** of street characteristics.（量化分析）
- 64-dimensional computational vector, where each dimension corresponds to the **cover ratio of a specific scene element** in the field of view (FOV), which **indicates the spatial area that is visible from a location**
- contributes to the calculations between place and other spatial, and socioeconomic variables.（贡献类）

### A human-machine adversarial scoring framework for urban perception assessment using street-view images
- Traditionally, **the evaluation of human perceptions towards their visual surroundings** remains difficult due to the lack of **high-throughput methods, inadequate sample problems and being restricted to interviews and questionnaires**（传统方法的缺陷）
- urban perception assessment process
- **multi-sources** of geospatial big data
- **massive geo-tagged** imagery datasets
- **intuitive way** for urban residents to gain perceptions about their surrounding environments （意义类）
- **tackle the large-scale derivation problem** for urban perception（大数据带来的问题）
- has emerged as a **promising data source to infer urban perceptions**（数据源）
-  Street-view imagery is **primarily distributed along urban streets** and represents the **physical morphological properties of urban interior spaces** （街景）
- **assess the effect of a city’s environment** on social and economic **outcomes**
- overcoming the **inadequate sample problem** and certain limits imposed by traditional interview and questionnaire approach（贡献类，客服传统数据的问题）
- **their special political and economic status and physical environments**（物理、社会经济环境两方面） 
- an urban perception is a **subjective assessment** and is influenced by people’s social and cultural backgrounds（主观性制约）
- **rapidly and costeffectively assess** local urban perceptions（修饰类、快速有效）
- This study developed a framework with deep learning, street-view imagery and iterative feedback mechanism and to assess **cityscale urban perceptions**（修饰类，城市尺度）
- We conducted a case study of an urban perception assessment in a **high-density urban environment**, e.g., Wuhan, to demonstrate **the efficacy of the proposed framework**. （案例、评估）
- we analyzed the driving factors to **explain the results from both the visual and urban functional aspects**.（两方面，可视化和城市功能）


