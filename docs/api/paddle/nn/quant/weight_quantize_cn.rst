.. _cn_api_paddle_nn_quant_weight_quantize:

weight_quantize
-------------------------------
.. py:function:: paddle.nn.quant.weight_quantize(x, algo='weight_only_int8', arch=None)

weight_only 和 llm.int8 权重的量化函数。

参数
::::::::::::
    - **x** (Tensor) - 待量化的输入张量，数据类型为 float16 或 bfloat16。
    - **algo** (str) - 应用于 x 的算法，必须是 '`weight_only_int8`'、'`weight_only_int4`' 和 '`llm.int8`' 中的一个，默认为 '`weight_only_int8`'。
    - **arch** (int) - 针对目标设备的计算架构。例如，A100 为 80，v100 为 70，如果您没有指定架构，我们将从您的设备获取架构，默认为 None。

返回
::::::::::::
    - **out** (Tensor) - 量化结果的张量，数据类型为 int8，形状为 x 的转置。
    - **scale** (Tensor) - 每个通道的比例张量，数据类型为 float32。

代码示例：
::::::::::

COPY-FROM: paddle.nn.quant.weight_quantize
