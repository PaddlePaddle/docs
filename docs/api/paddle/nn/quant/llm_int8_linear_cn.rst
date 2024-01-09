.. _cn_api_paddle_nn_quant_llm_int8_linear:

llm_int8_linear
-------------------------------

.. py:function:: paddle.nn.quant.llm_int8_linear(x, weight, bias=None, weight_scale=None, threshold=6.0)

应用两个张量的矩阵乘法。若提供了偏置，则进行偏置加法。

细节可参考论文 `LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale <https://arxiv.org/abs/2208.07339>`_ 。

此方法要求 CUDA 版本不低于 11.2。

参数
::::::::::::
    - **x** (Tensor) - 第一个输入张量，将被乘以，数据类型为 float16 或 bfloat16。
    - **weight** (Tensor) - 第二个输入张量，将被乘以。其秩必须为 2。
    - **bias** (Tensor|None) - 输入的偏置张量。如果为 None，则不执行偏置加法。否则，偏置将被加到矩阵乘法结果上。
    - **weight_scale** (Tensor|None) - 提供给权重的输入比例张量，用于反量化。其秩必须为 1。
    - **threshold** (float) - 激活中离群值的最小值，离群值的通道将应用与 x.dtype 的乘法。

返回
::::::::::::
    - ``Tensor``：输出张量，其数据类型与 x 相同。

返回类型
::::::::::::
Tensor

代码示例：
::::::::::

COPY-FROM: paddle.nn.quant.llm_int8_linear
