.. _cn_api_fluid_layers_zeros:

zeros
-------------------------------

.. py:function:: paddle.fluid.layers.zeros(shape,dtype,force_cpu=False)

该 OP 创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为 0 的 Tensor。

参数
::::::::::::

    - **shape** (tuple|list|Tensor) - 输出 Tensor 的形状，``shape`` 的数据类型为 int32 或者 int64。
    - **dtype** (np.dtype|str) - 输出 Tensor 的数据类型，数据类型必须为 bool、 float16、float32、float64、int32 或 int64。
    - **force_cpu** (bool，可选) - 是否强制将输出 Tensor 写入 CPU 内存。如果 ``force_cpu`` 为 False，则将输出 Tensor 写入当前所在运算设备的内存，默认为 False。

返回
::::::::::::
值全为 0 的 Tensor，数据类型和 ``dtype`` 定义的类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.zeros
