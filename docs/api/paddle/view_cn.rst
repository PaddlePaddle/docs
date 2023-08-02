.. _cn_api_paddle_tensor_view:

view
--------------------------------

.. py:function:: paddle.view(x, shape_or_dtype, name=None)

使用特定的 shape 或者 dtype，返回 x 的一个 view Tensor。

仅在动态图下可用，返回的 Tensor 和 x 共享内存。

参数
:::::::::

    - **x** (Tensor) - 输入多维 Tensor，可选的数据类型为 'float16'、'float32'、'float64'、'int16'、'int32'、'int64'、'bool'、'uint16'。
    - **shape_or_dtype** (list|tuple|np.dtype|str|VarType) - 指定的新的 shape 或者 dtype。如果是 list 或者 tuple，则代表 shape，否则代表 dtype。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，x 的一个 view Tensor。


代码示例
:::::::::

COPY-FROM: paddle.view
