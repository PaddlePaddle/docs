.. _cn_api_paddle_dataset_imikolov:

imikolov
-------------------------------

imikolov的简化版数据集。

此模块将从 http://www.fit.vutbr.cz/~imikolov/rnnlm/ 下载数据集，并将训练集和测试集解析为paddle reader creator。

.. py:function:: paddle.dataset.imikolov.build_dict(min_word_freq=50)

从语料库构建一个单词字典，字典的键是word，值是这些单词从0开始的ID。

.. py:function:: paddle.dataset.imikolov.train(word_idx, n, data_type=1)

imikolov训练数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个单词ID元组。

参数：
    - **word_idx** (dict) – 词典
    - **n** (int) – 如果类型是ngram，表示滑窗大小；否则表示序列最大长度
    - **data_type** (数据类型的成员变量(NGRAM 或 SEQ)) – 数据类型 (ngram 或 sequence)

返回： 训练数据集的reader creator

返回类型：callable

.. py:function::paddle.dataset.imikolov.test(word_idx, n, data_type=1)

imikolov测试数据集的creator。

它返回一个reader creator, reader中的每个样本的是一个单词ID元组。

参数：
    - **word_idx** (dict) – 词典
    - **n** (int) – 如果类型是ngram，表示滑窗大小；否则表示序列最大长度
    - **data_type** (数据类型的成员变量(NGRAM 或 SEQ)) – 数据类型 (ngram 或 sequence)

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.imikolov.convert(path)

将数据集转换为recordio格式。



