.. _cn_api_fluid_layers_pow:

pow
-------------------------------

.. py:function:: paddle.pow(x, exponent, name=None)




该OP是指数激活算子：

.. math::

    out = x^{exponent}

**注意：如果需要对输入进行 elementwise_pow 操作，请查使用** :ref:`cn_api_fluid_layers_elementwise_pow` 。

参数：
    - **x** （Variable）- 多维 ``Variable``，数据类型为 ``float32`` 或 ``float64`` 。
    - **exponent** （float32|Variable）- ``float32`` 或形状为[1]的 ``Variable``，数据类型为 ``float32``。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置。默认值： ``None``。

返回：维度与输入 `x` 相同的 ``Variable``，数据类型与 ``x`` 相同。

返回类型：Variable。


**代码示例：**

.. code-block:: python

            import paddle
            import numpy as np
            x = fluid.data(name="x", shape=[32,32], dtype="float32")
            paddle.enable_imperative()
            
            # example 1: exponent is a float
            x_data = np.array([1, 2, 3])
            exponent = 2
            x = paddle.imperative.to_variable(x_data)
            res = paddle.pow(x, exponent)
            print(res.numpy()) # [1 4 9]
            
            # example 2: exponent is a Variable
            exponent = paddle.fill_constant(shape=[1], value=2, dtype='float32')
            res = paddle.pow(x, exponent)
            print(res.numpy()) # [1 4 9]







