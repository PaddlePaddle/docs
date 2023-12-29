.. _cn_api_paddle_incubate_nn_functional_fused_dropout_add:

fused_dropout_add
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_dropout_add(x, y, p=0.5, training=True, mode='upscale_in_train', name=None)

实现了 Dropout 和 Add 的融合

参数
:::::::::
    - **x** (Tensor): 输入张量。数据类型为 bfloat16, float16, float32 或 float64.
    - **y** (Tensor): 输入张量，数据类型为 bfloat16, float16, float32 或 float64.

    - **p** (float|int, 可选): 将单位设置为零的概率。默认值: 0.5。
    - **training** (bool, 可选): 标记是否为训练阶段。默认值: True。
    - **mode**(str, 可选): ['upscale_in_train'(默认) | 'downscale_in_infer'].

            1. upscale_in_train (默认), 在训练时放大输出

                - 训练: :math:`out = x \times \frac{mask}{(1.0 - dropout\_prob)} + y`
                - 推理: :math:`out = x + y`

            2. downscale_in_infer, 在推理时缩小输出

                - 训练: :math:`out = input \times mask + y`
                - 推理: :math:`out = input \times (1.0 - dropout\_prob) + y`

    - **name** (str, 可选): 操作的名称, 默认值: None。有关更多信息, 请参阅 :ref:`api_guide_Name`.


返回
:::::::::
    - 融合 dropout 和 add 的张量具有与 `x` 相同的形状和数据类型。


代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_dropout_add
