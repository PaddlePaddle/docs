.. _cn_api_paddle_dataset_imdb:

imdb
-------------------------------

IMDB数据集。

本模块的数据集从 http://ai.stanford.edu/%7Eamaas/data/sentiment/IMDB 数据集。这个数据集包含了25000条训练用电影评论数据，25000条测试用评论数据，且这些评论带有明显情感倾向。此外，该模块还提供了用于构建词典的API。


.. py:function:: paddle.dataset.imdb.build_dict(pattern, cutoff)

从语料库构建一个单词字典，词典的键是word，值是这些单词从0开始的ID。


.. py:function:: paddle.dataset.imdb.train(word_idx)

IMDB训练数据集的creator。


它返回一个reader creator, reader中的每个样本的是一个从0开始的ID序列，标签范围是[0，1]。


参数：
    - **word_idx** (dict) – 词典

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imdb.test(word_idx)

IMDB测试数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个从0开始的ID序列，标签范围是[0，1]。

参数：
    - **word_idx** (dict) – 词典

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imdb.convert(path)

将数据集转换为recordio格式。


