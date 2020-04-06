.. _cn_api_fluid_io_Dataset:

Dataset
-------------------------------

.. py:class:: paddle.fluid.io.Dataset()

概述Dataset的方法和行为的抽象类。

映射式(map-style)数据集需要继承这个基类，映射式数据集为可以通过一个键值索引并获取指定样本的数据集，所有映射式数据集须实现以下方法：

``__getitem__`` : 根据给定索引获取数据集中指定样本，在 ``fluid.io.DataLoader`` 中需要使用此函数通过下标获取样本。

``__len__`` : 返回数据集样本个数， ``fluid.io.BatchSampler`` 中需求样本个数来生成下标序列。
