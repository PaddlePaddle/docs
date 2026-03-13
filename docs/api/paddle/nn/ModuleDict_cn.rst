.. _cn_api_paddle_nn_ModuleDict:

ModuleDict
-------------------------------

.. py:class:: paddle.nn.ModuleDict(modules=None)




ModuleDict 用于保存子层到有序字典中，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 有序字典一样被访问。

.. note::
   ``LayerDict`` 是 ``ModuleDict`` 的别名，两者在使用和功能上完全等价。

参数
::::::::::::

    - **modules** (ModuleDict|OrderedDict|list[(key, Module)]，可选) - 键值对的可迭代对象，值的类型为 `paddle.nn.Module` 。


代码示例
::::::::::::

COPY-FROM: paddle.nn.ModuleDict

方法
::::::::::::
clear()
'''''''''

清除 ModuleDict 中所有的子层。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.clear

pop()
'''''''''

移除 ModuleDict 中的键 并且返回该键对应的子层。

**参数**

    - **key** (str) - 要移除的 key。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.pop

keys()
'''''''''

返回 ModuleDict 中键的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.keys

items()
'''''''''

返回 ModuleDict 中键/值对的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.items


values()
'''''''''

返回 ModuleDict 中值的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.values


update()
'''''''''

更新子层中的键/值对到 ModuleDict 中，会覆盖已经存在的键。

**参数**

    - **modules** (ModuleDict|OrderedDict|list[(key, Module)]) - 键值对的可迭代对象，值的类型为 ``paddle.nn.Module`` 。

**代码示例**

COPY-FROM: paddle.nn.ModuleDict.update
