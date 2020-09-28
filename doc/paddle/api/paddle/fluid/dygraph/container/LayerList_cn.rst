.. _cn_api_fluid_dygraph_LayerList:

LayerList
-------------------------------

.. py:class:: paddle.fluid.dygraph.LayerList(sublayers=None)




LayerList用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规python列表一样被索引。

参数：
    - **sublayers** (iterable，可选) - 要保存的子层。

返回：无

**代码示例**

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
