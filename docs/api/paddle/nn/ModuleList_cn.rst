.. _cn_api_paddle_nn_ModuleList:

ModuleList
-------------------------------

.. py:class:: paddle.nn.ModuleList(sublayers=None)




ModuleList 用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 列表一样被索引。

.. note::
   ``ModuleList`` 是 ``LayerList`` 的别名，两者在使用和功能上完全等价。

参数
::::::::::::

    - **sublayers** (iterable，可选) - 要保存的子层。


代码示例
::::::::::::

COPY-FROM: paddle.nn.ModuleList

方法
::::::::::::
append(sublayer)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

添加一个子层到整个 list 的最后。

**参数**

    - **sublayer** (Module) - 要添加的子层。别名 ``module``。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.append

insert(index, sublayer)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

向 list 中插入一个子层，到给定的 index 前面。

**参数**

    - **index** (int) - 要插入的位置。
    - **sublayer** (Module) - 要插入的子层。别名 ``module``。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.insert

extend(sublayers)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

添加多个子层到整个 list 的最后。

**参数**

    - **sublayers** (iterable of Module) - 要添加的所有子层。别名 ``modules``。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.extend
