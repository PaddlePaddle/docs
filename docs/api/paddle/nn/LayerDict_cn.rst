.. _cn_api_nn_LayerDict:

LayerDict
-------------------------------

.. py:class:: paddle.nn.LayerDict(sublayers=None)




LayerDict 用于保存子层到有序字典中，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 有序字典一样被访问。

参数
::::::::::::

    - **sublayers** (LayerDict|OrderedDict|list[(key, Layer)]，可选) - 键值对的可迭代对象，值的类型为 `paddle.nn.Layer` 。


代码示例
::::::::::::

COPY-FROM: paddle.nn.LayerDict:code-example1

方法
::::::::::::
clear()
'''''''''

清除 LayerDict 中所有的子层。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:clear

pop()
'''''''''

移除 LayerDict 中的键 并且返回该键对应的子层。

**参数**

    - **key** (str) - 要移除的 key。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:pop

keys()
'''''''''

返回 LayerDict 中键的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:keys


items()
'''''''''

返回 LayerDict 中键/值对的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:items


values()
'''''''''

返回 LayerDict 中值的可迭代对象。

**参数**

    无。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:values


update()
'''''''''

更新子层中的键/值对到 LayerDict 中，会覆盖已经存在的键。

**参数**

    - **sublayers** (LayerDict|OrderedDict|list[(key, Layer)]) - 键值对的可迭代对象，值的类型为 `paddle.nn.Layer` 。

**代码示例**

COPY-FROM: paddle.nn.LayerDict:update
