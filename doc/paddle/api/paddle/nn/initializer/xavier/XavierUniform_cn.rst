.. _cn_api_nn_initializer_XavierUniform:

XavierUniform
-------------------------------

.. py:class:: paddle.nn.initializer.XavierUniform(fan_in=None, fan_out=None, name=None)


该类实现Xavier权重初始化方法（ Xavier weight initializer），Xavier权重初始化方法出自Xavier Glorot和Yoshua Bengio的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

该初始化函数用于保持所有层的梯度尺度几乎一致。

在均匀分布的情况下，取值范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in+fan\_out}}

参数：
    - **fan_in** (float，可选) - 用于Xavier初始化的fan_in，从tensor中推断。默认为None。
    - **fan_out** (float，可选) - 用于Xavier初始化的fan_out，从tensor中推断。默认为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：
    由使用均匀分布的Xavier权重初始化的参数。

**代码示例**：

.. code-block:: python

    import paddle

    data = paddle.ones(shape=[3, 1, 2], dtype='float32')
    weight_attr = paddle.framework.ParamAttr(
        name="linear_weight",
        initializer=paddle.nn.initializer.XavierUniform())
    bias_attr = paddle.framework.ParamAttr(
        name="linear_bias",
        initializer=paddle.nn.initializer.XavierUniform())
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    # linear.weight:  [[-0.04229349 -1.1248565 ]
    #                  [-0.10789523 -0.5938053 ]]
    # linear.bias:  [ 1.1983747  -0.40201235]

    res = linear(data)
    # res:  [[[ 1.0481861 -2.1206741]]
    #        [[ 1.0481861 -2.1206741]]
    #        [[ 1.0481861 -2.1206741]]]
