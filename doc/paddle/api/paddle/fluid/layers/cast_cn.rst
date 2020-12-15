.. _cn_api_fluid_layers_cast:

cast
-------------------------------

.. py:function:: paddle.cast(x,dtype)




该OP将 ``x`` 的数据类型转换为 ``dtype`` 并输出。支持输出和输入的数据类型相同。

参数：
    - **x** (Tensor) - 输入的多维Tensor或LoDTensor，支持的数据类型为：bool、float16、float32、float64、uint8、int32、int64。
    - **dtype** (str|np.dtype|core.VarDesc.VarType) - 输出Tensor的数据类型。支持的数据类型为：bool、float16、float32、float64、int8、int32、int64、uint8。

返回：Tensor或LoDTensor，维度与 ``x`` 相同，数据类型为 ``dtype``

返回类型：Tensor

**代码示例**：

.. code-block:: python

  import paddle

  x = paddle.to_tensor([2, 3, 4], 'float64')
  y = paddle.cast(x, 'uint8')
