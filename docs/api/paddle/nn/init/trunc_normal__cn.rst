.. _cn_api_paddle_nn_init_trunc_normal_:

trunc_normal\_
-------------------------------

.. py:class:: paddle.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

将输入张量的值设置为截断的正态分布的随机数，该操作会直接修改输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **mean** (float，可选) - 正态分布的均值，默认值为 0.0。
    - **std** (float，可选) - 正态分布的标准差，默认值为 1.0。
    - **a** (float，可选) - 截断的下限，默认值为-2.0。
    - **b** (float，可选) - 截断的上限，默认值为 2.0。
