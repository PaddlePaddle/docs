.. _cn_api_nn_initializer_TruncatedNormal:

TruncatedNormal
-------------------------------

.. py:class:: paddle.nn.initializer.TruncatedNormal(mean=0.0, std=1.0)




Random Truncated Normal(高斯)分布初始化函数

参数：
    - **mean** (float16|float32) - 正态分布的平均值
    - **std** (float16|float32) - 正态分布的标准差
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：对象

**代码示例**

.. code-block:: python

    import paddle

    data = paddle.ones(shape=[3, 1, 2], dtype='float32')
    weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
    bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=2.0))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    res = linear(data)
