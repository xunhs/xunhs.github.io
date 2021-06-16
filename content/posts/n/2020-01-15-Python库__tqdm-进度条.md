---
layout: post
cid: 73
title: Python库-tqdm-进度条
slug: 73
date: 2020-01-15T09:03:58+00:00
updated: '2020/04/13 17:58:48'
status: publish
author: Ethan
categories:
  - 收藏
tags:
  - progressbar
  - 进度条
  - tqdm
  - Python
abbrlink: bddb6b32
---



> 总结tqdm库，进度条显示。


<!--more-->


**t**e **q**uiero **d**e**m**asiado



### 基本用法
```Python
from tqdm import tqdm
for i in tqdm(range(10000)):
    ...

```

### Manual
```Python
pbar = tqdm(total=100)
for i in range(10):
    time.sleep(0.1)
    pbar.update(10)
pbar.close()

```
### Advanced 

```Python
from tqdm import tqdm 
bar = tqdm(ncm_list)
for ncm_item in bar:
    # print(ncm_item.stem)
    bar.set_description_str(desc=ncm_item.stem)
    dump(ncm_item)


'''
效果：
动态更新描述部分
'''
```

### Pandas Integrate

**df.progress_apply**

```Python
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))

# Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
# (can use `tqdm.gui.tqdm`, `tqdm.notebook.tqdm`, optional kwargs, etc.)
tqdm.pandas(desc="my bar!")

# Now you can use `progress_apply` instead of `apply`
# and `progress_map` instead of `map`
df.progress_apply(lambda x: x**2)
# can also groupby:
# df.groupby(0).progress_apply(lambda x: x**2)

```

### Keras Integration

```Python
from tqdm.keras import TqdmCallback

...

model.fit(..., verbose=0, callbacks=[TqdmCallback()])

```

### IPython/Jupyter Integration

```Python

from tqdm.notebook import trange, tqdm
from time import sleep

for i in trange(3, desc='1st loop'):
    for j in tqdm(range(100), desc='2nd loop', leave=False):
        sleep(0.01)

```

<!-- 正文内容 -->
***

<!-- 图片位置 -->

<!-- 图片位置 -->