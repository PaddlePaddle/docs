.. _cn_api_fluid_layers_hard_sigmoid:

hard_sigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)

sigmoid的分段线性逼近激活函数，速度比sigmoid快，详细解释参见 https://arxiv.org/abs/1603.00391。

.. math::

      \\out=\max(0,\min(1,slope∗x+offset))\\

参数：
    - **x** (Variable) - 该OP的输入为多维Tensor。数据类型必须为float32或float64。
    - **slope** (float，可选) - 斜率。值必须为正数，默认值为0.2。
    - **offset** (float，可选) - 偏移量。默认值为0.5。
    - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回：激活后的Tensor，形状、数据类型和 ``x`` 一致。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.fill_constant(shape=[3, 2], value=0.5, dtype='float32') # [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    result = fluid.layers.hard_sigmoid(data) # [[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]]
