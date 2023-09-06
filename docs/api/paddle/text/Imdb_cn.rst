.. _cn_api_text_datasets_Imdb:

Imdb
-------------------------------

.. py:class:: paddle.text.datasets.Imdb()


该类是对 `IMDB <https://www.imdb.com/interfaces/>`_ 测试数据集的实现。

参数
:::::::::
    - **data_file** (str) - 保存压缩数据的路径，如果参数 :attr:`download`设置为 True，可设置为 None。默认为 None。
    - **mode** (str) - 'train' 或'test' 模式。默认为'train'。
    - **cutoff** (int) - 构建词典的截止大小。默认为 Default 150。
    - **download** (bool) - 如果 :attr:`data_file`未设置，是否自动下载数据集。默认为 True。

返回
:::::::::
``Dataset``， IMDB 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.text.datasets.Imdb
