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

.. code-block:: python

    import paddle
    import numpy as np

    class MyLayer(paddle.nn.Layer):
        def __init__(self):
            super(MyLayer, self).__init__()
            self.linears = paddle.nn.LayerList(
                [paddle.nn.Linear(10, 10) for i in range(10)])

        def forward(self, x):
            # LayerList can act as an iterable, or be indexed using ints
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
            return x

方法
::::::::::::
append()
'''''''''

添加一个子层到整个 list 的最后。

**参数**

    - **sublayer** (Layer) - 要添加的子层。

**代码示例**

.. code-block:: python

    import paddle

    linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
    another = paddle.nn.Linear(10, 10)
    linears.append(another)
    print(len(linears))  # 11


insert()
'''''''''

向 list 中插入一个子层，到给定的 index 前面。

**参数**

    - **index ** (int) - 要插入的位置。
    - **sublayers** (Layer) - 要插入的子层。

**代码示例**

.. code-block:: python

    import paddle

    linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
    another = paddle.nn.Linear(10, 10)
    linears.insert(3, another)
    print(linears[3] is another)  # True

extend()
'''''''''

添加多个子层到整个 list 的最后。

**参数**

    - **sublayers** (iterable of Layer) - 要添加的所有子层。

**代码示例**

.. code-block:: python

    import paddle

    linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
    another_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(5)])
    linears.extend(another_list)
    print(len(linears))  # 15
    print(another_list[0] is linears[10])  # True
