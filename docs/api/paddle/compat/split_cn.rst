.. _cn_api_paddle_compat_split:

split
-------------------------------

.. py:function:: paddle.compat.split(tensor, split_size_or_sections, dim=0)


PyTorch 兼容的 :ref:`cn_api_paddle_split` 版本，允许了非整除的 ``split_size_or_sections`` 输入

使用前请详细参考：`【输入参数用法不一致】torch.split`_ 以确定是否使用此模块。

.. _【输入参数用法不一致】torch.split: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.split.html

.. note::
    此 API 遵循 ``torch.split`` 的函数签名和行为以实现 PyTorch 兼容。
    如需使用 Paddle 原生实现，请参考 :ref:`cn_api_paddle_split`

参数
:::::::::
       - **tensor** (Tensor) - 输入 N 维 Tensor，支持 bool、bfloat16、float16、float32、float64、uint8、int8、int32 或 int64 数据类型
       - **split_size_or_sections** (int|list|tuple) - 若为整数，则将 Tensor 均匀分割为指定大小的块，与 :ref:`cn_api_paddle_split` 不同，本 API 不要求此参数整除对应维度的通道数：非整除情况下输出元组的最后一个 tensor 对应维度将为余数大小，小于此值。若为列表/元组，则按指定尺寸分割，禁止使用负值（例如对 9 通道的维度，``[2,3,-1]`` 会被拒绝）
       - **dim** (int|Tensor, 可选) - 分割维度，可为整数或形状为[]的 0-D Tensor（数据类型需为 ``int32`` 或 ``int64``）。若 ``dim < 0``，则实际维度为 ``rank(x) + dim``。默认值：0

返回
:::::::::
tuple(Tensor)，分割后的 Tensor 元组


代码示例
:::::::::

COPY-FROM: paddle.compat.split
