.. _cn_api_paddle_nn_ModuleList:

ModuleList
-------------------------------

.. py:class:: paddle.nn.ModuleList(modules=None)




ModuleList 用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 列表一样被索引。

.. note::
   ``LayerList`` 是 ``ModuleList`` 的别名，两者在使用和功能上完全等价。

参数
::::::::::::

    - **modules** (iterable，可选) - 要保存的子层。


代码示例
::::::::::::

COPY-FROM: paddle.nn.ModuleList

方法
::::::::::::
append()
'''''''''

添加一个子层到整个 list 的最后。

**参数**

    - **modules** (Module) - 要添加的子层。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.append

insert()
'''''''''

向 list 中插入一个子层，到给定的 index 前面。

**参数**

    - **index** (int) - 要插入的位置。
    - **module** (Module) - 要插入的子层。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.insert

extend()
'''''''''

添加多个子层到整个 list 的最后。

**参数**

    - **modules** (iterable of Module) - 要添加的所有子层。

**代码示例**

COPY-FROM: paddle.nn.ModuleList.extend
