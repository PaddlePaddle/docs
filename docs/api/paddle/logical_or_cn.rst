.. _cn_api_fluid_layers_logical_or:

logical_or
-------------------------------

.. py:function:: paddle.logical_or(x, y, out=None, name=None)

该OP逐元素的对 ``X`` 和 ``Y`` 进行逻辑或运算。

.. math::
        Out = X || Y

.. note::
    ``paddle.logical_or`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数：
        - **x** （Tensor）- 输入的 `Tensor` ，支持的数据类型为bool, int8, int16, in32, in64, float32, float64。
        - **y** （Tensor）- 输入的 `Tensor` ，支持的数据类型为bool, int8, int16, in32, in64, float32, float64。
        - **out** （Tensor，可选）- 指定算子输出结果的 `Tensor` ，可以是程序中已经创建的任何Tensor。默认值为None，此时将创建新的Tensor来保存输出结果。
        - **name** （str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回： ``Tensor`` ， 维度``x`` 维度相同，存储运算后的结果。

**代码示例：**

.. code-block:: python

     import paddle
     import numpy as np

     x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
     y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
     x = paddle.to_tensor(x_data)
     y = paddle.to_tensor(y_data)
     res = paddle.logical_or(x, y)
     print(res) # [[True, True] [True, False]]
