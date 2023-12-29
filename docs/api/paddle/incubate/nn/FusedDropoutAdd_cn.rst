.. _cn_api_paddle_incubate_nn_FusedDropoutAdd:

FusedDropoutAdd
-------------------------------

.. py:class::paddle.incubate.nn. FusedDropoutAdd ( p=0.5, mode='upscale_in_train', name=None )

实现了 Dropout 和 Add 的融合

参数
:::::::::
    - **p** (float|int, 可选): 将单位设置为零的概率。默认值: 0.5
    - **mode** (str, 可选): ['upscale_in_train'(默认) | 'downscale_in_infer']

               1. upscale_in_train (默认), 在训练时放大输出

                  - train: :math:`out = x \times \frac{mask}{(1.0 - p)} + y`
                  - inference: :math:`out = x + y`

               2. downscale_in_infer, 在推理时缩小输出

                  - train: :math:`out = x \times mask + y`
                  - inference: :math:`out = x \times (1.0 - p) + y`
    - **name** (str, 可选): 操作的名称, 默认值为 None. 有关更多信息请参阅 :ref:`api_guide_Name`.


形状
:::::::::
        - x: N-D 张量.
        - y: N-D 张量.
        - output: N-D 张量, 和 `x` 形状相同.


代码示例
::::::::::

COPY-FROM: paddle.incubate.nn.FusedDropoutAdd
