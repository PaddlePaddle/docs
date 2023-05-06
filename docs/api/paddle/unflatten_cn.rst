.. _cn_api_paddle_tensor_unflatten:

unflatten
--------------------------------

.. py:function:: paddle.unflatten(x, shape, axis, name=None)

"将输入 Tensor 沿指定轴 axis 上的维度展成 shape 形状"。与 :ref:`cn_api_paddle_tensor_flatten` 是反函数。


参数
:::::::::

    - **x** (Tensor) - 输入多维 Tensor，可选的数据类型为 'float16'、'float32'、'float64'、'int16', 'int32'、'int64'、'bool'、'uint16'。
    - **shape** (list|tuple|Tensor) - 在指定轴上将该维度展成 ``shape``， 其中 ``shape`` 最多包含一个 -1，如果输入 ``shape`` 不包含 -1 ，则乘积应该等于 ``x.shape[axis]`` 大小。 数据类型为 ``int``。如果 ``shape`` 的类型是 ``list`` 或 ``tuple``，它的元素可以是整数或者形状为[]的 ``Tensor``。如果 ``shape`` 的类型是 ``Tensor``，则是 1-D 的 ``Tensor``。
    - **axis** (int) - 要展开的轴，作为 ``x.shape`` 的索引。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
返回
:::::::::
Tensor，沿指定轴将维度展开的后的 ``x``。


代码示例
:::::::::

COPY-FROM: paddle.unflatten
