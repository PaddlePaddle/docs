.. _cn_api_text_datasets_Movielens:

Movielens
-------------------------------

.. py:class:: paddle.text.datasets.Movielens()


该类是对 `Movielens 1-M <https://grouplens.org/datasets/movielens/1m/>`_
测试数据集的实现。

参数
:::::::::

    - **data_file** (str) - 保存压缩数据的路径，如果参数 :attr:`download`设置为 True，可设置为 None。默认为 None。
    - **mode** (str) - 'train' 或 'test' 模式。默认为'train'。
    - **test_ratio** (float) - 为测试集划分的比例。默认为 0.1。
    - **rand_seed** (int) - 随机数种子。默认为 0。
    - **download** (bool) - 如果 :attr:`data_file`未设置，是否自动下载数据集。默认为 True。

返回值
:::::::::
    ``Dataset``，Movielens 1-M 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.text.datasets.Movielens
