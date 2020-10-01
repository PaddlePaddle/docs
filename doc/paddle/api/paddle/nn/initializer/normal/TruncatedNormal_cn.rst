.. _cn_api_nn_initializer_TruncatedNormal:

TruncatedNormal
-------------------------------

.. py:class:: paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0)


随机截断正态（高斯）分布初始化函数。

参数
    - **mean** (float，可选) - 正态分布的平均值。默认值为 0。
    - **std** (float，可选) - 正态分布的标准差。默认值为 1.0。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：
    由随机截断正态（高斯）分布初始化的参数。

**代码示例**

.. code-block:: python

    import paddle

    data = paddle.ones(shape=[3, 1, 2], dtype='float32')
    weight_attr = paddle.framework.ParamAttr(
        name="linear_weight",
        initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
    bias_attr = paddle.framework.ParamAttr(
        name="linear_bias",
        initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    # linear.weight:  [[-1.0981836  1.4140984]
    #                  [ 3.1390522 -2.8266568]]
    # linear.bias:  [-2.1546738 -1.6570673]

    res = linear(data)
    # res:  [[[-0.11380529 -3.0696259 ]]
    #        [[-0.11380529 -3.0696259 ]]
    #        [[-0.11380529 -3.0696259 ]]
