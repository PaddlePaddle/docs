.. _cn_api_paddle_nn_functional_rms_norm:

rms_norm
-------------------------------

.. py:function:: paddle.nn.functional.rms_norm(x, normalized_shape, weight=None, epsilon=1e-05, name=None)

对输入 Tensor 的最后一维应用 RMS Layer Normalization，使用 CUDA 实现。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，形状为 [rows, cols] 或更高维（会被展平为 2 维），数据类型为 bfloat16、float16、float32 或 float64。
    - **normalized_shape** (list|tuple) - 期望输入的形状 :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`。如果是一个整数，会对最后一维进行归一化，该维度的大小需为指定值。
    - **weight** (Tensor，可选) - rms_norm 权重的 Tensor，默认为 None。
    - **epsilon** (float|None，可选) - 为了数值稳定加在分母上的小值。如果为 None，则使用计算类型的机器精度：``float64`` 输入使用 ``np.finfo(np.float64).eps``（双精度），其他所有类型使用 ``np.finfo(np.float32).eps``（单精度）。默认值：None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
    out (Tensor) - 与输入形状相同的归一化 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.rms_norm
