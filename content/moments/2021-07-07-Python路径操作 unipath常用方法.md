---
title: 'Python路径操作:unipath常用方法'
date: 2021-07-07T02:45:11.000Z

slug: python路径操作-unipath常用方法
tags:
    - Python
    - unipath
    - 路径操作
    - Path
---

<!-- content -->

{{< button href="[https://...](https://docs.python.org/3/library/pathlib.html)" >}}unipath{{< /button >}}

```python
from pathlib import Path

path = Path("../root")
new_path = path / "new_dir" / "file" # 路径拼接

# 基本属性:
path.suffix　　　　#文件后缀
path.stem　　　　　 #文件名不带后缀
path.name　　　　　　#带后缀的完整文件名
path.parent　　　　#路径的上级目录

# 基本函数:
path.iterdir()　　#遍历目录的子目录或者文件
path.glob(pattern)　　#过滤目录(返回生成器)
# -- example --
Path('.').glob('*.py') # 遍历py文件
# >>> [Path('pathlib.py'), Path('setup.py'), Path('test_pathlib.py')]
Path('.').glob('*/*.py')
# >>> [Path('docs/conf.py')]
Path('.').glob('**/*.py') # 递归遍历py文件
# >>> [Path('build/lib/pathlib.py'), Path('docs/conf.py'), 
# Path('pathlib.py'), Path('setup.py'), Path('test_pathlib.py')]
# ---

path.exists()　　#判断路径是否存在
path.is_dir()　　#判断是否是目录
path.is_file()  #判断是否是文件

path.mkdir(parents=False, exist_ok=False)　　#创建目录
# If parents is true, any missing parents of this path are created as needed
# If exist_ok is true, FileExistsError exceptions will be ignored

path.unlink()　　#删除文件或目录(目录非空触发异常)
path.open(mode='r', encoding=None)　　#打开文件(支持with)

path.with_name()　　#更改完整文件名称
path.with_suffix()　　#更改后缀
path.with_stem()    #更改文件名不带后缀
# -- example --
Path('c:/Downloads/draft.txt').with_stem('final')
# >>> Path('c:/Downloads/final.txt')
# ---

```
<!--more-->
