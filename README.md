retrival
========

关于bm25召回的包都是集成的, 太重了而且很慢, 所以自己写了个包用, 基于Numba加速会快很多

保留了一定的更新空间, 后续看看能不能加入(chao xi)更多的方法, 如bm25+之类的

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

corpus = [{"text": "我 是 一个 大 傻瓜",  "label": "A", "id": "0"},
{"text": "我 是 一个 小 小 猪", "label": "B", "id": "1"},
{"text": "我 是 一个 大 大 猪", "label": "C", "id": "2"},
{"text": "我 是 一个 大 小 猪", "label": "D", "id": "3"}]

path_jsonl = 'retrival.jsonl'

## 创建对象
se = SearchEngine(index_name='test', path_jsonl=path_jsonl)

## 基于corpus建立索引文件, 注意这里会把corpus写入对象
se.index(corpus=corpus)

## 基于bm25的召回
se.search('我是一个小猪', n_top=100)

## 重新生成jsonlines的索引文件
se.init_corpus_to_jsonl()
```
