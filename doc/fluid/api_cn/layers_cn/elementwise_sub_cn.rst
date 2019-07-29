.. _cn_api_fluid_layers_elementwise_sub:

elementwise_sub
-------------------------------

.. py:function:: paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)

逐元素相减算子

等式是：

.. math::
       Out = X - Y

- **X** ：任何维度的张量（Tensor）。
- **Y** ：维度必须小于或等于**X**维度的张量（Tensor）。

此运算算子有两种情况：
        1. :math:`Y` 的形状（shape）与 :math:`X` 相同。
        2. :math:`Y` 的形状（shape）是 :math:`X` 的连续子序列。

对于情况2：
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 将是 :math:`Y` 传到 :math:`X` 上的起始维度索引。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis = rank（X）-rank（Y）` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾随维度将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: text

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** （Tensor）- 第一个输入张量（Tensor）。
        - **y** （Tensor）- 第二个输入张量（Tensor）。
        - **axis** （INT）- （int，默认-1）。将Y传到X上的起始维度索引。
        - **act** （basestring | None）- 激活函数名称，应用于输出。
        - **name** （basestring | None）- 输出的名称。

返回：        元素运算的输出。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    # 例1: shape(x) = (2, 3, 4, 5), shape(y) = (2, 3, 4, 5)
    x0 = fluid.layers.data(name="x0", shape=[2, 3, 4, 5], dtype='float32')
    y0 = fluid.layers.data(name="y0", shape=[2, 3, 4, 5], dtype='float32')
    z0 = fluid.layers.elementwise_sub(x0, y0)
     
    # 例2: shape(X) = (2, 3, 4, 5), shape(Y) = (5)
    x1 = fluid.layers.data(name="x1", shape=[2, 3, 4, 5], dtype='float32')
    y1 = fluid.layers.data(name="y1", shape=[5], dtype='float32')
    z1 = fluid.layers.elementwise_sub(x1, y1)
     
    # 例3: shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    x2 = fluid.layers.data(name="x2", shape=[2, 3, 4, 5], dtype='float32')
    y2 = fluid.layers.data(name="y2", shape=[4, 5], dtype='float32')
    z2 = fluid.layers.elementwise_sub(x2, y2, axis=2)
     
    # 例4: shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    x3 = fluid.layers.data(name="x3", shape=[2, 3, 4, 5], dtype='float32')
    y3 = fluid.layers.data(name="y3", shape=[3, 4], dtype='float32')
    z3 = fluid.layers.elementwise_sub(x3, y3, axis=1)
     
    # 例5: shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    x4 = fluid.layers.data(name="x4", shape=[2, 3, 4, 5], dtype='float32')
    y4 = fluid.layers.data(name="y4", shape=[2], dtype='float32')
    z4 = fluid.layers.elementwise_sub(x4, y4, axis=0)
     
    # 例6: shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
    x5 = fluid.layers.data(name="x5", shape=[2, 3, 4, 5], dtype='float32')
    y5 = fluid.layers.data(name="y5", shape=[2], dtype='float32')
    z5 = fluid.layers.elementwise_sub(x5, y5, axis=0)





