---
layout: post
cid: 84
title: 阅读笔记-从Word Embedding到Bert模型
slug: 84
date: 2020-02-20T09:12:07+00:00
updated: '2020/02/22 11:55:38'
status: publish
author: Ethan
categories:
  - 收藏
tags:
  - Bert
  - Embedding
abbrlink: '37187034'
---


***

<!-- Abstract -->
>  [知乎文章](https://zhuanlan.zhihu.com/p/49271699)-阅读笔记：从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史

<!-- Abstract -->

<!--more-->

<!-- 正文内容 -->

目录

[TOC]

### 简介  
NLP中的预训练技术是一步一步如何发展到Bert模型的
- Bert的思路是如何逐渐形成的
- Bert的历史沿革是什么，继承了什么，创新了什么，为什么效果那么好，主要原因是什么
- 为何说模型创新不算太大，为何说Bert是近年来NLP重大进展的集大成者

### 先从图像领域的预训练说起
- Frozen
- Fine-Tuning
- 对于层级的CNN结构来说，不同层级的神经元学习到了不同类型的图像特征，由底向上特征形成层级结构，如上图所示，如果我们手头是个人脸识别任务，训练好网络后，把每层神经元学习到的特征可视化肉眼看一看每层学到了啥特征，你会看到最底层的神经元学到的是线段等特征，图示的第二个隐层学到的是人脸五官的轮廓，第三层学到的是人脸的轮廓，通过三步形成了特征的层级结构，**越是底层的特征越是所有不论什么领域的图像都会具备的比如边角线弧线等底层基础特征，越往上抽取出的特征越与手头任务相关**。正因为此，所以预训练好的网络参数，尤其是**底层的网络参数抽取出特征跟具体任务越无关，越具备任务的通用性，所以这是为何一般用底层预训练好的参数初始化新任务网络参数的原因。而高层特征跟任务关联较大，实际可以不用使用，或者采用Fine-tuning用新数据集合清洗掉高层无关的特征抽取器。**

### Word Embedding考古史
- 语言模型
- 神经网络语言模型(NNLM, 2003)
- Word2Vec(2013)
  - CBOW
  - Skip-gram
- Word Embedding后下游任务是怎么用它的
  - 句子中每个单词以Onehot形式作为输入，然后乘以学好的Word Embedding矩阵Q，就直接取出单词对应的Word Embedding
  - Word Embedding矩阵Q其实就是网络Onehot层到embedding层映射的网络参数矩阵
- 有什么问题值得改进的
  - Word Embedding其实对于很多下游NLP任务是有帮助的，只是帮助没有大到闪瞎忘记戴墨镜的围观群众的双眼而已
  - 多义词问题
  - 多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

### 从Word Embedding到ELMO
- 基于上下文的Embedding-ELMO: Embedding from Language Models ([From Deep contextualized word representation](https://arxiv.org/abs/1802.05365))
- 之前的Word Embedding本质上是个**静态的方式**，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的Word Embedding不会跟着上下文场景的变化而改变，所以对于比如Bank这个词，它事先学好的Word Embedding中混合了几种语义 ，在应用中来了个新句子，即使从上下文中（比如句子包含money等词）明显可以看出它代表的是“银行”的含义，但是对应的Word Embedding内容也不会变，它还是混合了多种语义。
- ELMO预训练
  - ELMO的本质思想是：我事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，不过这没关系。在我实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候我可以**根据上下文单词的语义去调整单词的Word Embedding表示**，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。
  - 每个编码器的深度都是**两层LSTM叠加**。这个网络结构其实在NLP中是很常用的。
  - 句子中**每个单词都能得到对应的三个Embedding**:最底层是单词的Word Embedding，往上走是第一层双向LSTM中对应单词位置的Embedding，这层编码单词的**句法信息**更多一些；再往上走是第二层LSTM中对应单词位置的Embedding，这层编码单词的**语义信息**更多一些。也就是说，ELMO的预训练过程不仅仅学会单词的Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。 
  - ![3mlhwt.jpg](https://s2.ax1x.com/2020/02/20/3mlhwt.md.jpg)
- 预训练好网络结构后，如何给下游任务使用呢？
  - 这样句子X中每个单词在ELMO网络中都能获得对应的三个Embedding，之后给予这三个Embedding中的每一个Embedding一个权重a，这个权重可以学习得来，根据各自权重累加求和，将三个Embedding整合成一个。然后将整合后的这个Embedding作为X句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用。
  - 这一类预训练的方法被称为“Feature-based Pre-Training”

- 多义词问题解决了么
  - 静态Word Embedding无法解决多义词的问题，那么ELMO引入上下文动态调整单词的embedding后多义词问题解决了吗？解决了，而且比我们期待的解决得还要好。
- ELMO有什么缺点？
  - ELMO使用了LSTM而不是新贵Transformer， Transformer提取特征的能力是要远强于LSTM的
  - ELMO采取双向拼接这种融合特征的能力可能比Bert一体化的融合特征方式弱
  - ELMO是基于特征融合的预训练方法

### 从Word Embedding到GPT 
- 生成式预训练-GPT：Generative Pre-Training (第一个阶段是利用语言模型进行预训练，第二阶段通过Fine-tuning的模式解决下游任务)
- Transformer
  - Transformer是个叠加的“自注意力机制（Self Attention）”构成的深度网络，是目前NLP里最强的特征提取器，注意力这个机制在此被发扬光大，从任务的配角不断抢戏，直到Transformer一跃成为踢开RNN和CNN传统特征提取器，荣升头牌，大红大紫。
  - [深度学习中的注意力模型](https://zhuanlan.zhihu.com/p/37601161) - 补充下相关基础知识
  - Transformer比较好的文章可以参考以下两篇文章
    - [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)可视化地介绍Transformer
    - 哈佛大学NLP研究组写的[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 
    - Transformer在未来会逐渐替代掉RNN成为主流的NLP工具，RNN一直受困于其并行计算能力，这是因为它本身结构的序列性依赖导致的，尽管很多人在试图通过修正RNN结构来修正这一点，但是我不看好这种模式，因为**给马车换轮胎不如把它升级到汽车**，这个道理很好懂，更何况目前汽车的雏形已经出现了，干嘛还要执着在换轮胎这个事情呢？是吧？再说CNN，CNN在NLP里一直没有形成主流，CNN的最大优点是易于做并行计算，所以速度快，但是在捕获NLP的序列关系尤其是长距离特征方面天然有缺陷，不是做不到而是做不好，目前也有很多改进模型，但是特别成功的不多。综合各方面情况，很明显**Transformer同时具备并行性好，又适合捕获长距离特征**，没有理由不在赛跑比赛中跑不过RNN和CNN。
- GPT训练好了如何使用
  - 结构改造 （这里慢慢就没看懂了）
  - 对网络参数进行Fine-tuning
- GPT的缺点
  - 要是把语言模型改造成双向的就好了

### Bert的诞生
- NLP的四大类任务
  - ![3mGg4P.jpg](https://s2.ax1x.com/2020/02/20/3mGg4P.md.jpg)
  - 序列标注
    - 中文分词，词性标注，命名实体识别，语义角色标注等
    - 特点是句子中每个单词要求模型根据上下文都要给出一个分类类别
  - 分类任务
    - 文本分类，情感计算等
    - 特点是不管文章有多长，总体给出一个分类类别即可
  - 句子关系判断
    - Entailment，QA，语义改写，自然语言推理等
    - 特点是给定两个句子，模型判断出两个句子是否具备某种语义关系
  - 生成式任务
    - 机器翻译，文本摘要，写诗造句，看图说话等
    - 特点是输入文本内容后，需要自主生成另外一段文字
- Bert普适性
- Bert如何改造下有任务？
- Bert效果如何
  - 在11个各种类型的NLP任务中达到目前最好的效果，某些任务性能有极大的提升。
- 从GPT和ELMO及Word2Vec到Bert：四者的关系
  - 如果我们把GPT预训练阶段换成双向语言模型，那么就得到了Bert；而如果我们把ELMO的特征抽取器换成Transformer，那么我们也会得到Bert。所以你可以看出：Bert最关键两点，一点是特征抽取器采用Transformer；第二点是预训练的时候采用双向语言模型。
  - Bert：最近几年NLP重要技术的集大成者
- Bert如何改造双向语言模型？
  - Masked LM
  - Next Sentence Prediction

- Bert评价及总结
  - ![3mlhwt.jpg](https://s2.ax1x.com/2020/02/20/3mY1oR.md.jpg)
  - Bert借鉴了ELMO，GPT及CBOW，主要提出了Masked 语言模型及Next Sentence Prediction，但是这里Next Sentence Prediction基本不影响大局，而Masked LM明显借鉴了CBOW的思想。所以说Bert的模型没什么大的创新，更像最近几年NLP重要进展的集大成者
  - 首先是两阶段模型，第一阶段双向语言模型预训练，这里注意要用双向而不是单向，第二阶段采用具体任务Fine-tuning或者做特征集成；第二是特征抽取要用Transformer作为特征提取器而不是RNN或者CNN；第三，双向语言模型可以采取CBOW的方法去做
  - Bert最大的亮点在于效果好及普适性强，几乎所有NLP任务都可以套用Bert这种两阶段解决思路，而且效果应该会有明显提升。可以预见的是，未来一段时间在NLP应用领域，Transformer将占据主导地位，而且这种两阶段预训练方法也会主导各种应用。



<!-- 正文内容 -->
***

<!-- 图片位置 -->


<!-- 图片位置 -->

