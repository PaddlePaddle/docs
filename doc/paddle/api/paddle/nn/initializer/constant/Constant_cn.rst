.. _cn_api_nn_initializer_Constant:

Constant
-------------------------------

.. py:class:: paddle.nn.initializer.Constant(value=0.0)




该接口为常量初始化函数，用于权重初始化，通过输入的value值初始化输入变量；

参数：
        - **value** (float16|float32) - 用于初始化输入变量的值；

返回：参数初始化类的实例

**代码示例**

.. code-block:: python

    import paddle
    import paddle.nn as nn

    data = paddle.rand([30, 10, 2], dtype='float32')
    linear = nn.Linear(2, 4, weight_attr=nn.initializer.Constant(value=2.0))
    res = linear(data)
    print(linear.weight.numpy())
    #result is [[2. 2. 2. 2.],[2. 2. 2. 2.]]
