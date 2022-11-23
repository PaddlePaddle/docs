.. _cn_api_fluid_dygraph_LayerList:

LayerList
-------------------------------

.. py:class:: paddle.nn.LayerList(sublayers=None)




LayerList 用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 列表一样被索引。

参数
::::::::::::

    - **sublayers** (iterable，可选) - 要保存的子层。


代码示例
::::::::::::

COPY-FROM: paddle.nn.LayerList

方法
::::::::::::
append()
'''''''''

添加一个子层到整个 list 的最后。

**参数**

    - **sublayer** (Layer) - 要添加的子层。

**代码示例**

COPY-FROM: paddle.nn.LayerList.append


insert()
'''''''''

向 list 中插入一个子层，到给定的 index 前面。

**参数**

    - **index** (int) - 要插入的位置。
    - **sublayers** (Layer) - 要插入的子层。

**代码示例**

COPY-FROM: paddle.nn.LayerList.insert

extend()
'''''''''

添加多个子层到整个 list 的最后。

**参数**

    - **sublayers** (iterable of Layer) - 要添加的所有子层。

**代码示例**

COPY-FROM: paddle.nn.LayerList.extend
