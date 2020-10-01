.. _cn_api_nn_initializer_Assign:

Assign
-------------------------------

.. py:class:: paddle.nn.initializer.Assign(value)




该OP使用Numpy型数组来初始化参数变量。

参数：
    - **value** （numpy） - 用于初始化变量的一个Numpy型数组、Python列表、Paddle Tensor。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：张量（Tensor）

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # numpy array
    data = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2,2])))
    bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2])))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    res = linear(data)

    # python list
    data = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign([2,2]))
    bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign([2]))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    res = linear(data)

    # paddle tensor
    data = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
    bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    res = linear(data)
