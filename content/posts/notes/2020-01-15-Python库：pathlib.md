---
layout: post
cid: 72
title: Python库：pathlib
slug: 72
date: 2020-01-18T09:04:38+00:00
updated: '2020/02/21 15:20:16'
status: publish
author: Ethan
categories:
  - 收藏
tags:
  - Python
  - pathlib
  - 文件/文件夹
abbrlink: 891ae564
---


***

<!-- Abstract -->
> pathlib 是 python3 非常好用的文件/文件夹操作的库，总结常用用法，参考自[jianshu](https://www.jianshu.com/p/a820038e65c3)

<!-- Abstract -->

<!--more-->

<!-- 正文内容 -->

### 常用操作
```Python
from pathlib import Path
p = Path()

p = Path(r'd:\test\tt.txt.bk')
p.name # 获取文件名
# tt.txt.bk
p.stem # 获取文件名除后缀的部分
# tt.txt
p.suffix # 文件后缀
# .bk
p.suffixs # 文件的后缀们...
# ['.txt', '.bk']
p.parent # 相当于dirnanme
# WindowsPath('d:/test')
p.parents # 返回一个iterable, 包含所有父目录
# <WindowsPath.parents>
for i in p.parents:
  print(i)
# d:\test
# d:\
a.parts # 将路径通过分隔符分割成一个元祖
# ('d:\\', 'test', 'tt.txt.bk')


p = Path(p, 'tt.txt') # 字符串拼接
p.exists() # 判断文件是否存在
p.is_file() # 判断是否是文件
p.is_dir() # 判断是否是目录

```
### 遍历文件夹
```Python
p = Path(r'd:\test')
# WindowsPath('d:/test')
p.iterdir() # 相当于os.listdir
p.glob('*') # 相当于os.listdir, 但是可以添加匹配条件
p.rglob('*') # 递归遍历
```
### 创建文件夹
```Python
p = Path(r'd:\test\tt\dd')
p.mkdir(exist_ok=True) # 创建文件目录(前提是tt目录存在, 否则会报错)
# 一般我会使用下面这种创建方法
p.mkdir((exist_ok=True, parents=True) # 递归创建文件目录

```




<!-- 正文内容 -->
***

<!-- 图片位置 -->

<!-- 图片位置 -->