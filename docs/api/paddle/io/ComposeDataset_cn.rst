.. _cn_api_io_ComposeDataset:

ComposeDataset
-------------------------------

.. py:class:: paddle.io.ComposeDataset

由多个数据集的字段组成的数据集。

这个数据集用于将多个映射式（map-style）且长度相等数据集按字段组合为一个新的数据集。

参数
::::::::::::

    - **datasets** (list of Dataset) - 待组合的多个数据集。

返回
::::::::::::
Dataset，字段组合后的数据集。

代码示例
::::::::::::

COPY-FROM: paddle.io.ComposeDataset
