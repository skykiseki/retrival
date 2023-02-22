import jieba
import numba as nb
import json
import numpy as np
import jsonlines
import os
from indxr import Indxr
from numba.typed import List as TypedList
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Iterable, Dict
from .utils import bm25


class SearchEngine(object):
    """


    Attributes:
    ----------
    index_name: str, 索引名称

    stop_words: list, 停用词表

    path_jsonl: str, 建立索引生成的jsonl文件

    inverted_index: dict, docs相关tf统计

    doc_lens: int, 总长度

    relative_doc_lens: float, 相对长度

    avg_doc_len: float, 语料的平均长度

    vocabulary: set, 词袋集合

    doc_count: int, 训练用的语料条数

    id_mapping: dict, 原始语料的id

    doc_index: Indxr object, 原始doc索引

    corpus: list[dict], 原始语料
    """

    def __init__(self, index_name, path_jsonl='retrival.jsonl', stop_words=None):
        if stop_words is None:
            stop_words = []
        self.index_name = index_name
        self.stop_words = stop_words
        self.path_jsonl = path_jsonl

        self.inverted_index = None
        self.doc_lens = None
        self.relative_doc_lens = None
        self.avg_doc_len = None
        self.vocabulary = None
        self.doc_count = None
        self.id_mapping = None
        self.doc_index = None

        self.corpus = None

    def query_preprocessing(self, query: str, method='search') -> List[str]:
        """
        query文本的预处理

        Parameters:
        ----------
        query: str, 字符串

        method: str, default/search模式

        Returns:
        -------
        terms: list[str], 返回切分后的词汇列表

        """
        assert method in ['search', 'default']

        if method == 'search':
            terms = [word for word in jieba.lcut_for_search(query) if word not in self.stop_words]
        else:
            terms = [word for word in jieba.lcut(query) if word not in self.stop_words]

        return terms

    def build_inverted_index(self, collections: Iterable) -> Dict:
        """

        Parameters:
        ----------
        collections: 语料

        Returns:
        -------
        self

        """
        vectorizer = CountVectorizer(dtype=np.int16, token_pattern=r"(?u)\b\w+\b")

        # [n_docs x n_terms]
        df_matrix = vectorizer.fit_transform(raw_documents=collections)

        # [n_terms x n_docs]
        df_matrix = df_matrix.transpose().tocsr()

        # 抓一下词袋
        vocabulary = vectorizer.get_feature_names_out()

        # 计算词袋矩阵
        inverted_index = defaultdict(dict)

        for i, term in enumerate(vocabulary):
            inverted_index[term]["doc_ids"] = df_matrix[i].indices
            inverted_index[term]["tfs"] = df_matrix[i].data

        self.inverted_index = dict(inverted_index)

        # 计算每段语料的长度
        self.doc_lens = np.squeeze(np.asarray(df_matrix.sum(axis=0), dtype=np.float32))

        # 计算相对长度（后续计算会用）
        self.relative_doc_lens = self.doc_lens / np.mean(self.doc_lens, dtype=np.float32)

        # 计算语料规模
        self.doc_count = df_matrix.shape[1]

        # 计算docs的平均长度
        self.avg_doc_len = np.mean(self.doc_lens, dtype=np.float32)

        # 计算词袋集合
        self.vocabulary = set(self.inverted_index)

    def index(self, corpus):
        """
        创建索引序列, 注意这里jsonl必须要有text字段和id字段

        corpus e.g.:
        [{'text': '1 2 3', 'id': '0'},
        {'text': 'a s d', 'id': '99'}]

        Parameters:
        ----------
        corpus: list[dict], 原始语料

        Returns:
        -------
        self

        """
        # 判断是否写入语料
        self.corpus = corpus

        # 先基于语料生成jsonl文件
        self.init_corpus_to_jsonl()

        # 创建text的generator
        collections = self.read_jsonl(path_jsonl=self.path_jsonl, col='text')

        # 创建id的id_mapping
        ids = self.read_jsonl(path_jsonl=self.path_jsonl, col='id')
        self.id_mapping = dict(enumerate(ids))

        # 取得原始语料
        self.doc_index = Indxr(self.path_jsonl)

        # 对collections进行建立索引、计算长度、相对长度等
        self.build_inverted_index(collections=collections)

    def get_doc_ids(self, query_terms: List[str]) -> nb.types.List:
        """

        Parameters:
        ----------
        query_terms: list, 切词后的列表

        Returns:
        -------
        每个word击中的doc id（numba转化后的嵌套列表）

        """
        return TypedList([self.inverted_index[t]["doc_ids"] for t in query_terms])

    def get_term_doc_freqs(self, query_terms: List[str]) -> nb.types.List:
        """

        Parameters:
        ----------
        query_terms: list, 切词后的列表

        Returns:
        -------
        每个word击中的doc 词频（numba转化后的嵌套列表）

        """
        return TypedList([self.inverted_index[t]["tfs"] for t in query_terms])

    @staticmethod
    def read_jsonl(path_jsonl, col) -> Iterable:
        """
        读取jsonline返回一个迭代器, 注意这里jsonl必须要有text, id字段

        Parameters:
        ----------
        path_jsonl: str, jsonline文件路径

        col: str, yield返回的字段名称

        Returns:
        -------
        Iterable, 一个迭代器
        """
        with open(path_jsonl, 'r') as fr:
            for l in fr:
                l = json.loads(l)
                yield l[col]

    def init_corpus_to_jsonl(self):
        """
        读取语料重新生成jsonl文件

        Parameters:
        ----------

        Returns:
        -------

        """
        # 先删除原始文件
        if os.path.isfile(self.path_jsonl):
            os.remove(self.path_jsonl)

        # 再开始生成jsonl文件
        if self.corpus is None:
            raise NotImplementedError('No corpus data found')

        else:
            corpus_writer = jsonlines.open(self.path_jsonl, mode='a')
            for d in self.corpus:
                corpus_writer.write(d)

            corpus_writer.close()


    def get_docs(self, doc_ids: List[str]) -> List[dict]:
        """
        基于原始id获取原始docs

        Parameters:
        ----------
        doc_ids: list, 原始docs的id

        Returns:
        -------
        list[dict], 最终召回的docs

        """
        return self.doc_index.mget(doc_ids)

    def search(self, query,
               n_top=100,
               b=0.75,
               k1=1.2):
        """

        Parameters:
        ----------
        query: str, 字符串

        n_top: 返回排序后的N个

        b, k1: float, 都是公式参数

        Returns:
        -------


        """
        query_terms = self.query_preprocessing(query)

        # 注意只取在词袋中的词汇
        query_terms = [t for t in query_terms if t in self.vocabulary]

        if len(query_terms) == 0:
            return {}

        # 返回击中的doc ids, 注意这里返回的是列表嵌套列表
        doc_ids = self.get_doc_ids(query_terms)

        # 返回关联的语料词频, 注意这里返回的是列表嵌套列表
        term_doc_freqs = self.get_term_doc_freqs(query_terms)

        # 按bm25开始召回, 这里跟上面doc_ids不一样的是, 这里有做倒排筛选, 同时这里是用的enumerate产生的自增序列id
        unique_doc_ids, scores = bm25(doc_ids=doc_ids,
                                      term_doc_freqs=term_doc_freqs,
                                      relative_doc_lens=self.relative_doc_lens,
                                      doc_count=self.doc_count,
                                      n_top=n_top,
                                      b=b,
                                      k1=k1)

        # 取得原始doc的id
        unique_doc_ids = [self.id_mapping[doc_id] for doc_id in unique_doc_ids]

        # 整理输出结果
        results = []
        docs_re = self.get_docs(unique_doc_ids)
        for doc, score in zip(docs_re, scores):
            doc["score"] = score
            results.append(doc)

        return results
