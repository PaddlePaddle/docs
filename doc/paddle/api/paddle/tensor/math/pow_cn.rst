.. _cn_api_paddle_tensor_math_pow:

pow
-------------------------------

.. py:function:: paddle.pow(x, y, name=None)



该OP是指数激活算子：

.. math::

    out = x^{y}

**注意：如果需要对输入进行 elementwise_pow 操作，请查使用** :ref:`cn_api_fluid_layers_elementwise_pow` 。

参数：
    - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``float32`` 或 ``float64`` 。
    - **y** （float32|Tensor）- ``float32`` 或形状为[1]的 ``Tensor``，数据类型为 ``float32``。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置。默认值： ``None``。

返回：维度与输入 `x` 相同的 ``Tensor``，数据类型与 ``x`` 相同。

返回类型：Tensor。


**代码示例：**

.. code-block:: python

            import paddle
            import numpy as np
            
            
            # example 1: y is a float
            x_data = np.array([1, 2, 3])
            y = 2
            x = paddle.to_tensor(x_data)
            res = paddle.pow(x, y)
            print(res.numpy()) # [1 4 9]
            
            # example 2: y is a Tensor
            x_data = np.array([1, 2, 3])
            y_data = np.array([2, 2, 2])

            x = paddle.to_tensor(x_data)
            y = paddle.to_tensor(y_data)

            res = paddle.pow(x, y)
            print(res.numpy()) # [1 4 9]







