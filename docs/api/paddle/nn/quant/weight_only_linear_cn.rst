.. _cn_api_paddle_nn_quant_weight_only_linear:

weight_only_linear
-------------------------------

.. py:function:: paddle.nn.quant.weight_only_linear(x, weight, bias=None, weight_scale=None, weight_dtype='int8', arch=None)

应用两个张量的矩阵乘法。若提供了偏置，则进行偏置加法。

此方法要求 CUDA 版本不低于 11.2。

参数
::::::::::::
    - **x** (Tensor) - 第一个输入张量，将被乘以，数据类型为 float16 或 bfloat16。
    - **weight** (Tensor) - 第二个输入张量，将被乘以。其秩必须为 2。
    - **bias** (Tensor|None) - 输入的偏置张量。如果为 None，则不执行偏置加法。否则，偏置将被加到矩阵乘法结果上。
    - **weight_scale** (Tensor|None) - 提供给权重的输入比例张量，用于反量化。其秩必须为 1。
    - **weight_dtype** (str) - 权重张量的数据类型，必须是 'int8', 'int4' 之一，默认为 'int8'。
    - **arch** (int) - 针对目标设备的计算架构。例如，A100 为 80，v100 为 70，如果您没有指定架构，我们将从您的设备获取架构，默认为 None。

返回
::::::::::::
    - ``Tensor``：输出张量，其数据类型与 x 相同。

代码示例：
::::::::::

COPY-FROM: paddle.nn.quant.weight_only_linear
