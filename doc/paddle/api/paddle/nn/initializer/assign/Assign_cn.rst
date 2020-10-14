.. _cn_api_nn_initializer_Assign:

Assign
-------------------------------

.. py:class:: paddle.nn.initializer.Assign(value, name=None)


该OP使用Numpy数组、Python列表、Tensor来初始化参数。

参数：
    - **value** （Tensor|numpy.ndarray|list） - 用于初始化参数的一个Numpy数组、Python列表、Tensor。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：
    由Numpy数组、Python列表、Tensor初始化的参数。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    # numpy array
    data_1 = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr_1 = paddle.framework.ParamAttr(
        name="linear_weight_1", 
        initializer=paddle.nn.initializer.Assign(np.array([2, 2])))
    bias_attr_1 = paddle.framework.ParamAttr(
        name="linear_bias_1",
        initializer=paddle.nn.initializer.Assign(np.array([2])))
    linear_1 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
    # linear_1.weight:  [2. 2.]
    # linear_1.bias:  [2.]

    res_1 = linear(data_1)
    # res_1:  [6.]

    # python list
    data_2 = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr_2 = paddle.framework.ParamAttr(
        name="linear_weight_2",
        initializer=paddle.nn.initializer.Assign([2, 2]))
    bias_attr_2 = paddle.framework.ParamAttr(
        name="linear_bias_2",
        initializer=paddle.nn.initializer.Assign([2]))
    linear_2 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
    # linear_2.weight:  [2. 2.]
    # linear_2.bias:  [2.]

    res_2 = linear(data_2)
    # res_2:  [6.]

    # tensor
    data_3 = paddle.ones(shape=[1, 2], dtype='float32')
    weight_attr_3 = paddle.framework.ParamAttr(
        name="linear_weight_3",
        initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
    bias_attr_3 = paddle.framework.ParamAttr(
        name="linear_bias_3",
        initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
    linear_3 = paddle.nn.Linear(2, 2, weight_attr=weight_attr_3, bias_attr=bias_attr_3)
    # linear_3.weight:  [2. 2.]
    # linear_3.bias:  [2.]

    res_3 = linear(data_3)
    # res_3:  [6.]

