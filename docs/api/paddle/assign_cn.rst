.. _cn_api_paddle_tensor_creation_assign:

assign
-------------------------------

.. py:function:: paddle.assign(x, output=None)




将输入数据拷贝至输出 Tensor。

参数
::::::::::::

    - **x** (Tensor|np.ndarray|list|tuple|scalar) - 输入 Tensor，或 numpy 数组，或由基本数据组成的 list/tuple，或基本数据，支持数据类型为 float16、float32、float64、int32、int64 和 bool。注意：由于当前框架的 protobuf 传输数据限制，float64 数据会被转化为 float32 数据。
    - **output** (Tensor，可选) - 输出 Tensor。如果为 None，则创建一个新的 Tensor 作为输出 Tensor。默认值为 None。

返回
::::::::::::
Tensor，形状、数据类型和值与 ``x`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.assign
