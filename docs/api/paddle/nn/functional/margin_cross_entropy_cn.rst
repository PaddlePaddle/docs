.. _cn_api_paddle_nn_functional_margin_cross_entropy:

margin_cross_entropy
-------------------------------

.. py:function:: paddle.nn.functional.margin_cross_entropy(logits, label, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, group=None, return_softmax=False, reduction='mean')

.. math::
    L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

其中，:math:`\theta_{y_i}` 是特征 :math:`x` 与类 :math:`w_{i}` 的角度。更详细的介绍请参考 ``Arcface loss``，https://arxiv.org/abs/1801.07698 。

提示：

    这个 API 支持单卡，也支持多卡（模型并行），使用模型并行时，``logits.shape[-1]`` 在每张卡上可以不同。

参数
::::::::::::

    - **logits** (Tensor) - 2-D Tensor，维度为 ``[N, local_num_classes]``，``logits`` 为归一化后的 ``X`` 与归一化后的 ``W`` 矩阵乘得到，数据类型为 float16，float32 或者 float64。如果用了模型并行，则 ``logits == sahrd_logits``。
    - **label** (Tensor) - 维度为 ``[N]`` 或者 ``[N, 1]`` 的标签。
    - **margin1** (float，可选) - 公式中的 ``m1``。默认值为 ``1.0``。
    - **margin2** (float，可选) - 公式中的 ``m2``。默认值为 ``0.5``。
    - **margin3** (float，可选) - 公式中的 ``m3``。默认值为 ``0.0``。
    - **scale** (float，可选) - 公式中的 ``s``。默认值为 ``64.0``。
    - **group** (Group，可选) - 通信组的抽象描述，具体可以参考 ``paddle.distributed.collective.Group``。默认值为 ``None``。
    - **return_softmax** (bool，可选) - 是否返回 ``softmax`` 概率值。默认值为 ``None``。
    - **reduction** （str，可选）- 是否对 ``loss`` 进行归约。可选值为 ``'none'`` | ``'mean'`` | ``'sum'``。如果 ``reduction='mean'``，则对 ``loss`` 进行平均，如果 ``reduction='sum'``，则对 ``loss`` 进行求和，``reduction='None'``，则直接返回 ``loss``。默认值为 ``'mean'``。

返回
::::::::::::

    - ``Tensor`` (``loss``) 或者 ``Tensor`` 二元组 (``loss``, ``softmax``) - 如果 ``return_softmax=False`` 返回 ``loss``，否则返回 (``loss``, ``softmax``)。当使用模型并行时 ``softmax == shard_softmax``，否则 ``softmax`` 的维度与 ``logits`` 相同。如果 ``reduction == None``，``loss`` 的维度为 ``[N, 1]``，否则为 ``[]``。

代码示例
::::::::::::
COPY-FROM: paddle.nn.functional.margin_cross_entropy:code-example1
COPY-FROM: paddle.nn.functional.margin_cross_entropy:code-example2
