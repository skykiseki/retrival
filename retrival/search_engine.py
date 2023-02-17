import os
import jieba


class SearchEngine(object):
    """


    Attributes:
    ----------
    index_name: str, 索引名称

    stop_words: list, 停用词表

    """
    def __init__(self, index_name, stop_words):
        self.index_name = index_name
        self.stop_words = stop_words

    def query_preprocessing(self, query):
        """
        query文本的预处理

        Parameters:
        ----------
        query: str, 字符串

        Returns:
        -------
        terms: list[str], 返回切分后的词汇列表

        """
        terms = [word for word in jieba.lcut_for_search(query) if word not in self.stop_words]

        return terms

    def search(self, query):
        """

        Parameters:
        ----------
        query: str, 字符串

        Returns:
        -------


        """
        query_terms = self.query_preprocessing(query)

        if len(query_terms) == 0:
            return {}

        return 1

