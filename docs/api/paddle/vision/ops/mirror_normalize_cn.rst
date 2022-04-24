.. _cn_api_paddle_vision_ops_mirror_normalize:

mirror_normalize
-------------------------------

.. py:function:: paddle.vision.ops.mirror_normalize(x, mirror, mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375], name=None)

对输入张量做随机水平翻转和归一化，将第一个维度视为批量大小，并根据输入张量中每个样本做随机水平翻转和归一化

.. note::
  此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - **x** (Tensor) - 形状为[N, ...]的输入Tensor，N为批次大小。
    - **mirror** (Tensor) - 形状为[N, 1]的输入Tensor，N为批次大小，并且每个值用户标识是否对该样本做水平翻转。
    - **mean** (float | List[float], 可选) - 每个通道归一化的平均值，默认为 [123.675, 116.28, 103.53]。
    - **std** (float | List[float], 可选) - 每个通道归一化的标准差，默认为 [58.395, 57.120, 57.375]。
    - **name** (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    与输入Tensor相同形状的输出Tensor，数据类型为float32

代码示例
:::::::::

COPY-FROM: <paddle.vision.ops.mirror_normalize>:<code-example>
