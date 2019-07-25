.. _cn_api_fluid_layers_l2_normalize:

l2_normalize
-------------------------------

.. py:function:: paddle.fluid.layers.l2_normalize(x,axis,epsilon=1e-12,name=None)

L2正则（L2 normalize Layer）

该层用欧几里得距离之和对维轴的x归一化。对于1-D张量（系数矩阵的维度固定为0），该层计算公式如下：

.. math::

    y=\frac{x}{\sqrt{\sum x^{2}+epsion}}

对于x多维的情况，该函数分别对维度轴上的每个1-D切片单独归一化

参数：
    - **x** (Variable|list)- l2正则层（l2_normalize layer）的输入
    - **axis** (int)-运用归一化的轴。如果轴小于0，归一化的维是rank(X)+axis。-1是最后维
    - **epsilon** (float)-epsilon用于避免分母为0，默认值为1e-12
    - **name** (str|None)-该层名称（可选）。如果设为空，则自动为该层命名

    返回：输出张量，同x的维度一致

    返回类型：变量

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="data",
                         shape=(3, 17, 13),
                         dtype="float32")
    normed = fluid.layers.l2_normalize(x=data, axis=1)









