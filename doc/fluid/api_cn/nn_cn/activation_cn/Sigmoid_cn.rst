.. _cn_api_nn_activate_Sigmoid:
Sigmoid

-------------------------------

.. py:class:: paddle.nn.loss.Sigmoid(name=None)

该接口用于创建一个 ``MarginRankingLoss`` 的可调用类。 这个类可以计算输入 `x` 经过激活函数 `sigmoid` 之后的值。

    .. math::

        output = \frac{1}{1 + e^{-input}}

参数
::::::::
  - **name** (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
::::::::
  - **x**(Tensor) - N-D tensor, 可以支持的数据类型是float16，float32，float64。

返回
::::::::
  返回计算 ``Sigmoid`` 的可调用对象。


**代码示例**

.. code-block:: python

     import numpy as np
     import paddle.fluid as fluid
     import paddle.imperative as imperative
     
     input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
     sigmoid = paddle.nn.layer.Sigmoid()
     x = imperative.to_variable(input_data)
     output = sigmoid(x)
     print(output.numpy()) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
