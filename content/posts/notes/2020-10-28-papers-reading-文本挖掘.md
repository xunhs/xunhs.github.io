---
title: Papers Reading-文本挖掘
author: Ethan
type: post
date: 2020-10-28T09:31:21+00:00
url: /2020/10/28/104/
argon_hide_readingtime:
  - 'false'
argon_meta_simple:
  - 'false'
argon_first_image_as_thumbnail:
  - default
views:
  - 5
categories:
  - 收藏
tags:
  - 文本挖掘
  - 论文阅读

---

> 与地理知识发现相关的文本挖掘论文阅读总结

<!--more-->

### 2017 - Extracting and analyzing semantic relatedness between cities using news articles

- Yingjie Hu
- [International Journal of Geographical Information Science](https://www.tandfonline.com/toc/tgis20/current)

- https://www.tandfonline.com/doi/abs/10.1080/13658816.2017.1367797?journalCode=tgis20

#### 札记

- Abstract短小精悍，逻辑性很强
- Introduction部分写的蛮好，**advantages 和 potential applications  部分做撰写参考**
- 创新点在于这个 computational framework；其框架里的方法我觉得还是蛮容易理解的

#### Abstract

- **news articles** reflect socioeconomic activities  + public concerns that <u>exist only in the perceptions of people</u>  
- **cities** are frequently mentioned in **news articles**, and two or more cities may **co-occur** (共现) in the same article.   
- **co-occurrence** => **relatedness** between cities => be under different topics
- **semantic relatedness**: the relatedness under different topics  
- By **reading news articles,** one can **grasp** the general **semantic relatedness between cities**  
- given <u>hundreds of thousands of news articles</u>, it is very difficult for anyone to <u>**manually** read them</u>.   
- proposes a **computational** framework:
  - <u>extract the semantic relatedness between cities</u>  
  - based on a natural language processing model  and employs a machine learning process to identify the main **topics**  
  - more than **500,000** news articles covering the **top 100 US cities** spanning a **10-year** period.   
  - perform exploratory **visualizations** of the extracted semantic relatedness under <u>different topics and over multiple years</u>  
  - analyze the impact of **geographic distance** on semantic relatedness and find varied distance decay effects  

#### Introduction

- news articles
  - rich information
  - timely
  - published online  
  - => becomes **useful data resource** for answering scientific questions  
- cities in news articles
  - cities are frequently **mentioned** in news articles.  => relatedness  
  - Cities can be related under a variety of topics.   => similarity / dissimilarity  
  - cities are interrelated into a **network**, in which the **nodes** are cities and the **edges** can have different semantics  
  - By reading news articles, one can **grasp** the general semantic relatedness between cities  
  - given hundreds of thousands of news articles, it is very difficult for anyone to **manually** read them.   
- the notion - semantic relatedness   
  - semantic relatedness in NLP
    - the words cat and mouse  
    - most people would <u>consider them as related</u>  
    - there exist films, books, and personal stories that <u>link these two words together</u>  
  - => **the semantic relatedness between cities**  
- **advantages** in using news articles for extracting semantic relatedness between cities  
  - accessed relatively **easily**   
  - capture **diverse** city relations 
  - enables a **temporal** exploration  
  - discover the **intangible** (抽象的，无形的) city relatedness perceived by people  
- potential **applications**   
  - In city planning and policy-making  
  - In geographic information retrieval (GIR)  
  - integrated with existing research on place-based GIS  
  - There also exist other possible applications  
- contributions  
  - a computational framework  
  - varied distance decay effects of the semantic relatedness  
  - perform exploratory **visualizations** on the multiple city networks；explore the **temporal** variations of the semantic relatedness between cities over years  

#### Method and Results

- 其实我理解就三个过程(Orz)：提取topic+建立语义关联+时空分析

![](https://i.loli.net/2020/10/29/tzlwmhNpy1xKcA6.png)

- 4,950-by-17 matrix
  - each row representing one city pair
  - each column representing one IPTC topic
  - cell containing the **relatedness value** of a city pair under a topic  
- two exploratory visualizations
  - based on the topics => construct 17 city networks based on the 17 IPTC topics  
  - based on the years  => obtain 10 matrices, each of which is 4,950 by 17 and contains the semantic relatedness in each year  
- the extracted semantic relatedness also **opens the door** to many other research questions, which can be grouped from spatial, temporal, and thematic perspectives.  
  - distance decay analysis  

***

<!-- 插入图片 -->
![](https://i.loli.net/2020/10/28/RhviTglJESIUQ4d.jpg)