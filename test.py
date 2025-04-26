from retrival import  SearchEngine


if __name__=="__main__":

    corpus = [
    {"text": "我 是 一个 小 小 猪", "label": "B", "id": "1"},
    {"text": "我 是 一个 大 大 猪", "label": "C", "id": "2"},
    {"text": "我 是 一个 大 小 猪", "label": "D", "id": "3"},
    {"text": "我有个手机，还是iphone，是jobs发明的。how are you?","label": "E", "id": "4"}
    ]

    path_jsonl = 'retrival.jsonl'

    ## 创建对象
    se = SearchEngine(index_name='test', path_jsonl=path_jsonl)

    ## 基于corpus建立索引文件, 注意这里会把corpus写入对象
    se.index(corpus=corpus)

    ## 基于bm25的召回
    result = se.search('我想要个iPhone', n_top=100)
    print(result)

    ## 重新生成jsonlines的索引文件
    # se.init_corpus_to_jsonl()