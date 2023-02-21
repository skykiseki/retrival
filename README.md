retrival
========

关于bm25召回的包都是集成的, 太重了而且很慢, 所以自己写了个包用, 基于Numba加速会快很多

安装说明
========

```shell
## 注意python版本需要>=3.8
pip install retrival
```

使用方法
========

```python
## 加载
from retrival import SearchEngine

## 创建对象
se = SearchEngine('test')

## 加载jsonlines文件
se.index('test.jsonl')

## 基于bm25的召回
se.search('我是一个小猪', n_top=100)
```

jsonlines文件示例:
------------------

数据示例:

```python
## 注意, 这里的text和id是必须的, 而且text必须以空格分隔的切词形式写入
{"text": "我 是 一个 大 傻瓜",  "label": "A", "id": "0"}
{"text": "我 是 一个 小 小 猪", "label": "B", "id": "1"}
{"text": "我 是 一个 大 大 猪", "label": "C", "id": "2"}
{"text": "我 是 一个 大 小 猪", "label": "D", "id": "3"}
```
