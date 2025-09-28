.. _cn_api_paddle_text_Imikolov:

Imikolov
-------------------------------

.. py:class:: paddle.text.Imikolov(data_file=None, data_type='NGRAM', window_size=-1, mode='train', min_word_freq=50, download=True)


该类是对 imikolov 测试数据集的实现。

参数
:::::::::

    - **data_file** (str) - 保存数据的路径，如果参数 :attr:`download`设置为 True，可设置为 None。默认为 None。
    - **data_type** (str) - 'NGRAM'或'SEQ'。默认为'NGRAM'。
    - **window_size** (int) - 'NGRAM'数据滑动窗口的大小。默认为-1。
    - **mode** (str) - 'train' 'test' mode. Default 'train'。
    - **min_word_freq** (int) - 构建词典的最小词频。默认为 50。
    - **download** (bool) - 如果 :attr:`data_file`未设置，是否自动下载数据集。默认为 True。

返回
:::::::::
``Dataset``，imikolov 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.text.Imikolov
